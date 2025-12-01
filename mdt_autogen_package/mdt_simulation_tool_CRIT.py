# mdt_autogen_package/mdt_simulation.py
import os
import glob
import json
import traceback # For detailed error logging
import autogen
from autogen import AssistantAgent, UserProxyAgent, GroupChat, GroupChatManager

# Import config from the package
from . import config

# Import Langchain RAG components for knowledge base
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader

# Global variable for knowledge base vector store
KNOWLEDGE_VECTORSTORE_GLOBAL = None

def create_mock_knowledge_base_files(kb_dir: str, guidelines_file_path: str, similar_cases_file_path: str):
    """
    Creates mock knowledge base files if they don't exist.
    """
    os.makedirs(kb_dir, exist_ok=True)
    if not os.path.exists(guidelines_file_path) and config.MOCK_GUIDELINE_CONTENT:
        with open(guidelines_file_path, "w", encoding="utf-8") as f:
            f.write(config.MOCK_GUIDELINE_CONTENT.strip())
        print(f"Created mock guidelines file: {guidelines_file_path}")

    if not os.path.exists(similar_cases_file_path) and config.MOCK_SIMILAR_CASES_CONTENT:
        with open(similar_cases_file_path, "w", encoding="utf-8") as f:
            f.write(config.MOCK_SIMILAR_CASES_CONTENT.strip())
        print(f"Created mock similar cases file: {similar_cases_file_path}")

def setup_knowledge_base_rag(kb_dir: str, guidelines_file_path: str, similar_cases_file_path: str) -> bool:
    """
    Sets up the RAG knowledge base using FAISS and Ollama embeddings.
    """
    global KNOWLEDGE_VECTORSTORE_GLOBAL
    
    try:
        print("Setting up Knowledge Base RAG for Chair agent...")
        create_mock_knowledge_base_files(kb_dir, guidelines_file_path, similar_cases_file_path)
        
        documents = [doc for file_path in [guidelines_file_path, similar_cases_file_path] if os.path.exists(file_path) for doc in TextLoader(file_path, encoding='utf-8').load()]
        
        if not documents:
            print("No documents found for knowledge base.")
            return False
        
        texts = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(documents)
        embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL_NAME, base_url=config.pro6000)
        KNOWLEDGE_VECTORSTORE_GLOBAL = FAISS.from_documents(texts, embeddings)
        
        print(f"Knowledge Base RAG setup completed. Loaded {len(texts)} text chunks.")
        return True
        
    except Exception as e:
        print(f"Failed to set up Knowledge Base RAG: {e}")
        traceback.print_exc()
        KNOWLEDGE_VECTORSTORE_GLOBAL = None
        return False

def query_knowledge_base_autogen_tool(query: str) -> str:
    """
    Tool function for AutoGen agents to query the knowledge base using RAG.
    """
    global KNOWLEDGE_VECTORSTORE_GLOBAL
    if KNOWLEDGE_VECTORSTORE_GLOBAL is None:
        return "Knowledge base is not available."
    try:
        relevant_docs = KNOWLEDGE_VECTORSTORE_GLOBAL.similarity_search(query, k=3)
        return "Relevant information from knowledge base:\n" + "\n\n".join([f"{i+1}. {doc.page_content}" for i, doc in enumerate(relevant_docs)]) if relevant_docs else "No relevant information found in knowledge base."
    except Exception as e:
        return f"Error querying knowledge base: {str(e)}."

def load_and_validate_agent_configs():
    """
    Scans for agent JSON configs, loads, validates them, and separates them into roles.
    """
    prompt_dir = os.path.join(os.path.dirname(__file__), '..', 'role_player_prompt')
    all_configs = []
    config_files = glob.glob(os.path.join(prompt_dir, '*.json'))

    if not config_files:
        raise FileNotFoundError(f"Configuration Error: No agent JSON files found in '{prompt_dir}'.")

    for config_file in sorted(config_files):
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            if 'agent_name' not in cfg or 'prompt' not in cfg:
                print(f"Warning: Skipping '{os.path.basename(config_file)}' due to missing 'agent_name' or 'prompt' key.")
                continue
            all_configs.append(cfg)
        except json.JSONDecodeError:
            print(f"Warning: Skipping '{os.path.basename(config_file)}' due to invalid JSON format.")
        except Exception as e:
            print(f"Warning: An unexpected error occurred while loading '{os.path.basename(config_file)}': {e}")
    
    planner_configs = [cfg for cfg in all_configs if cfg.get('is_planner')]
    participant_configs = [cfg for cfg in all_configs if not cfg.get('is_planner')]
    chair_configs = [cfg for cfg in participant_configs if cfg.get('is_chair')]

    if len(planner_configs) != 1:
        raise ValueError(f"Configuration Error: Expected 1 Planner agent ('is_planner': true), but found {len(planner_configs)}.")
    if len(chair_configs) != 1:
        raise ValueError(f"Configuration Error: Expected 1 Chair agent ('is_chair': true), but found {len(chair_configs)}.")

    return planner_configs[0], participant_configs

def plan_discussion_order(planner_prompt: str, patient_data_summary: str, participant_roles: list) -> (list, str):
    """
    Uses a PlannerAgent to decide the speaking order of all participants.
    """
    print("\n--- Phase 1: Planning Discussion Order ---")
    default_order = sorted(participant_roles)
    
    full_planner_prompt = f"""{planner_prompt}

The available specialist roles for ordering are: {participant_roles}. Your `order` array must contain exactly these strings, rearranged.
Patient Summary:
{patient_data_summary}
"""
    try:
        planner_agent = AssistantAgent(name="MDT_Planner", system_message="You generate valid JSON based on user instructions.", llm_config=config.llm_config_autogen)
        reply = planner_agent.generate_reply(messages=[{"role": "user", "content": full_planner_prompt}])
        
        reply_json_str = reply[reply.find('{'):reply.rfind('}')+1]
        parsed_reply = json.loads(reply_json_str)

        if all(k in parsed_reply for k in ['order', 'reasoning']) and isinstance(parsed_reply['order'], list):
            if sorted(parsed_reply['order']) == sorted(participant_roles):
                print(f"發言順序規劃原因：{parsed_reply['reasoning']}")
                return parsed_reply['order'], parsed_reply['reasoning']
        
        print("Warning: Planner response was invalid or did not contain all required roles. Reverting to default order.")
        return default_order, "Planner failed to provide a valid plan."
    except Exception as e:
        print(f"Warning: Could not get plan from planner (Error: {e}). Reverting to default order.")
        return default_order, "Planner failed due to an exception."

def run_autogen_mdt_simulation(patient_json_data: dict, patient_data_summary: str) -> str:
    """
    Runs a three-phase (Plan, Discuss, Summarize) AutoGen MDT simulation.
    """
    patient_id = patient_json_data.get("病歷號", "UnknownID")
    patient_name = patient_json_data.get("姓名", "N/A")
    print(f"\n--- Starting AutoGen MDT Meeting Simulation for Patient: {patient_name} (ID: {patient_id}) ---")

    try:
        planner_config, participant_configs = load_and_validate_agent_configs()
        participant_roles = [cfg['agent_name'] for cfg in participant_configs]
        chair_config = next((cfg for cfg in participant_configs if cfg.get('is_chair')), None)
    except (FileNotFoundError, ValueError) as e:
        print(f"Halting simulation due to configuration error: {e}")
        return f"Error: {e}"

    # Phase 1: Plan
    speaking_order, _ = plan_discussion_order(planner_config['prompt'], patient_data_summary, participant_roles)
    print(f"Planned speaking order: {speaking_order}")

    llm_config_no_tools = {**config.llm_config_autogen}
    llm_config_with_tools = {**config.llm_config_autogen, "tools": [{"type": "function", "function": {
        "name": "query_knowledge_base_autogen_tool", "description": "Query the medical knowledge base for ILD guidelines.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The medical query"}},"required": ["query"]}}}]}

    agent_objects = {}
    for cfg in participant_configs:
        is_chair = cfg.get('is_chair', False)
        agent = AssistantAgent(
            name=cfg['agent_name'],
            system_message=cfg['prompt'],
            llm_config=llm_config_with_tools if is_chair else llm_config_no_tools,
            max_consecutive_auto_reply=cfg.get('max_consecutive_auto_reply', 5 if is_chair else 3),
            function_map={"query_knowledge_base_autogen_tool": query_knowledge_base_autogen_tool} if is_chair else None
        )
        agent_objects[cfg['agent_name']] = agent
    
    ordered_agents = [agent_objects[role] for role in speaking_order if role in agent_objects]
    
    # --- Phase 2: Discussion ---
    print(f"\n--- Phase 2: Executing MDT Discussion (One Round) ---")
    print(f"Agents will speak in the following order: {[agent.name for agent in ordered_agents]}")

    discussion_groupchat = GroupChat(agents=ordered_agents, messages=[], max_round=len(ordered_agents) + 2, speaker_selection_method='round_robin', allow_repeat_speaker=False)
    manager = GroupChatManager(groupchat=discussion_groupchat, llm_config={**config.llm_config_autogen, "temperature": 0.1})
    
    chair_agent = agent_objects[chair_config['agent_name']]
    initial_prompt = f"Welcome to the ILD MDT meeting for patient {patient_name} (ID: {patient_id}). Please provide your initial assessment when it is your turn.\n\nPatient Summary:\n{patient_data_summary}\n\nLet's begin."
    
    try:
        print(f"Initiating discussion round...")
        # A user proxy agent is needed to kick off the chat and let the round_robin start
        initiator = UserProxyAgent(name="Initiator", code_execution_config=False, human_input_mode="NEVER")
        initiator.initiate_chat(manager, message=initial_prompt, clear_history=True)
        print(f"Discussion round completed. Total messages: {len(discussion_groupchat.messages)}")
    except Exception as e:
        print(f"Error during MDT discussion round: {e}")
        traceback.print_exc()
        return f"Error occurred during MDT discussion: {str(e)}"
    
    discussion_history = discussion_groupchat.messages

    # --- Phase 3: Summarization ---
    print("\n--- Phase 3: Generating Final Summary ---")
    final_summary = ""
    try:
        summary_prompt = """Based on the entire discussion history provided above, please act as the Chair (Rheumatologist) and provide a final synthesis.
Your summary MUST address the following seven questions:
1. Is this a case of ILD? (Yes/No/Uncertain)
2. Is Usual Interstitial Pneumonia (UIP) the predominant pattern? (Yes/No/Uncertain)
3. Is there a Non-specific Interstitial Pneumonia (NSIP) pattern? (Yes/No/Uncertain)
4. What is the final Connective Tissue Disease (CTD)-ILD assessment (type and activity)?
5. Is this a case of Progressive Fibrosing ILD (PF-ILD)? (Yes/No/Uncertain)
6. Should we adjust the patient's immunosuppression? (Provide a brief recommendation)
7. Should we recommend an anti-fibrotic agent? (Yes/No/Uncertain)

Conclude your entire response with the phrase 'MDT meeting concluded. TERMINATE_MDT_DISCUSSION_NOW' on a new line.
"""
        # Make a final, single call to the chair agent for summarization
        final_summary = chair_agent.generate_reply(messages=discussion_history + [{"role": "user", "content": summary_prompt}])
        print("Final summary generated successfully.")
    except Exception as e:
        print(f"Error during summarization phase: {e}")
        final_summary = f"Error during summarization: {e}"

    # Combine transcripts
    full_transcript = "--- DISCUSSION TRANSCRIPT ---\n"
    for msg in discussion_history:
        full_transcript += f"\n--- {msg.get('name', 'Unknown')} ---\n{msg.get('content', '')}\n"
    full_transcript += "\n--- FINAL SUMMARY ---\n"
    full_transcript += f"\n--- {chair_agent.name} ---\n{final_summary}\n"
    
    print(f"\n--- AutoGen MDT Simulation Concluded for Patient ID: {patient_id} ---")
    return full_transcript