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
    Returns three distinct items: the planner config, a list of specialist configs, and the chair config.
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
    chair_configs = [cfg for cfg in all_configs if cfg.get('is_chair')]
    # Specialists are participants who are NOT the planner and NOT the chair
    specialist_configs = [cfg for cfg in all_configs if not cfg.get('is_planner') and not cfg.get('is_chair')]

    if len(planner_configs) != 1:
        raise ValueError(f"Configuration Error: Expected 1 Planner agent ('is_planner': true), but found {len(planner_configs)}.")
    if len(chair_configs) != 1:
        raise ValueError(f"Configuration Error: Expected 1 Chair agent ('is_chair': true), but found {len(chair_configs)}.")
    if not specialist_configs:
        raise ValueError("Configuration Error: No specialist agents found (agents that are not planners or chairs).")

    return planner_configs[0], specialist_configs, chair_configs[0]


def plan_discussion_order(planner_prompt: str, patient_data_summary: str, specialist_roles: list) -> (list, str):
    """
    Uses a PlannerAgent to decide the speaking order of all participants.
    """
    print("\n--- Phase 1: Planning Discussion Order ---")
    default_order = sorted(specialist_roles)
    
    full_planner_prompt = f"""{planner_prompt}

The available specialist roles for ordering are: {specialist_roles}. Your `order` array must contain exactly these strings, rearranged.
Patient Summary:
{patient_data_summary}
"""
    try:
        planner_agent = AssistantAgent(name="MDT_Planner", system_message="You generate valid JSON based on user instructions.", llm_config=config.llm_config_autogen)
        reply = planner_agent.generate_reply(messages=[{"role": "user", "content": full_planner_prompt}])
        
        reply_json_str = reply[reply.find('{'):reply.rfind('}')+1]
        parsed_reply = json.loads(reply_json_str)

        if all(k in parsed_reply for k in ['order', 'reasoning']) and isinstance(parsed_reply['order'], list):
            if sorted(parsed_reply['order']) == sorted(specialist_roles):
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
        planner_config, specialist_configs, chair_config = load_and_validate_agent_configs()
        specialist_roles = [cfg['agent_name'] for cfg in specialist_configs]
        
    except (FileNotFoundError, ValueError) as e:
        print(f"Halting simulation due to configuration error: {e}")
        return f"Error: {e}"

    # Phase 1: Plan
    speaking_order, _ = plan_discussion_order(planner_config['prompt'], patient_data_summary, specialist_roles)
    print(f"Planned speaking order: {speaking_order}")

    llm_config_no_tools = {**config.llm_config_autogen}
    llm_config_with_tools = {**config.llm_config_autogen, "tools": [{"type": "function", "function": {
        "name": "query_knowledge_base_autogen_tool", "description": "Query the medical knowledge base for ILD guidelines.",
        "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "The medical query"}},"required": ["query"]}}}]}

    agent_objects = {}
    
    # Create specialist agents
    for cfg in specialist_configs:
        agent = AssistantAgent(
            name=cfg['agent_name'],
            system_message=cfg['prompt'],
            llm_config=llm_config_with_tools, # All specialists can use tools now
            max_consecutive_auto_reply=cfg.get('max_consecutive_auto_reply', 3), # Specialists have max 3 replies
            function_map={"query_knowledge_base_autogen_tool": query_knowledge_base_autogen_tool}
        )
        agent_objects[cfg['agent_name']] = agent
        
    # Create the chair agent separately
    chair_agent = AssistantAgent(
        name=chair_config['agent_name'],
        system_message=chair_config['prompt'],
        llm_config=llm_config_with_tools, # Chair can also use tools for summarization if needed
        max_consecutive_auto_reply=chair_config.get('max_consecutive_auto_reply', 5), # Chair has max 5 replies
        function_map={"query_knowledge_base_autogen_tool": query_knowledge_base_autogen_tool}
    )
    # The chair is not added to agent_objects for discussion round

    # Create ordered list of specialists for discussion
    ordered_specialist_agents = [agent_objects[role] for role in speaking_order if role in agent_objects]
    
    # --- Custom Speaker Selection Logic ---
    def custom_speaker_selection_logic(last_speaker: autogen.Agent, groupchat: autogen.GroupChat):
        """
        Custom speaker selection logic to handle tool calls correctly in a round-robin fashion.
        Ensures the agent that made a tool call gets the next turn to process the tool's output.
        """
        messages = groupchat.messages
        
        # Rule 1: If the last message is a tool response, find the original caller.
        if len(messages) > 1 and messages[-1].get("role") == "tool":
            # The tool response message in this AutoGen version is quirky. 
            # The tool_call_id is nested inside the 'tool_responses' list.
            if "tool_responses" in messages[-1] and messages[-1]["tool_responses"]:
                tool_call_id = messages[-1]["tool_responses"][0].get("tool_call_id")
                if tool_call_id:
                    # Search backwards for the assistant message that made the tool call
                    for i in range(len(messages) - 2, -1, -1):
                        msg = messages[i]
                        if msg.get("role") == "assistant" and "tool_calls" in msg and msg["tool_calls"]:
                            for tool_call in msg["tool_calls"]:
                                if tool_call.get("id") == tool_call_id:
                                    caller_name = msg.get("name")
                                    if caller_name in agent_objects:
                                        return agent_objects[caller_name]
        
        # Rule 2: Handle the initial turn after the Initiator.
        if last_speaker.name == "Initiator":
            return agent_objects[speaking_order[0]]

        # Rule 3: Normal round-robin progression.
        try:
            current_idx = speaking_order.index(last_speaker.name)
            next_idx = (current_idx + 1)
            # If we've reached the end of the round, terminate the discussion.
            if next_idx >= len(speaking_order):
                return None 
            return agent_objects[speaking_order[next_idx]]
        except (ValueError, IndexError):
            # Fallback: if the last speaker isn't in our planned specialist list, start from the beginning.
            return agent_objects[speaking_order[0]]

    # --- Phase 2: Discussion ---
    print(f"\n--- Phase 2: Executing MDT Discussion (One Round) ---")
    print(f"Specialist agents will speak in the following order: {[agent.name for agent in ordered_specialist_agents]}")

    discussion_groupchat = GroupChat(
        agents=ordered_specialist_agents, 
        messages=[], 
        max_round=15, # A safe upper limit for rounds
        speaker_selection_method=custom_speaker_selection_logic,
        allow_repeat_speaker=True # Crucial for allowing agent to speak after tool use
    )
    manager = GroupChatManager(groupchat=discussion_groupchat, llm_config={**config.llm_config_autogen, "temperature": 0.1})
    
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
        summary_prompt = chair_agent.system_message # Chair's system message already contains the summary prompt

        # Make a final, single call to the chair agent for summarization
        # Pass the discussion history and the summary prompt from the chair's system message
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