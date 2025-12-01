# mdt_autogen_package/mdt_simulation.py
import os
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
        
        # Create mock files if they don't exist
        create_mock_knowledge_base_files(kb_dir, guidelines_file_path, similar_cases_file_path)
        
        # Load documents
        documents = []
        for file_path in [guidelines_file_path, similar_cases_file_path]:
            if os.path.exists(file_path):
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
        
        if not documents:
            print("No documents found for knowledge base.")
            return False
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_documents(documents)
        
        # Create embeddings
        embeddings = OllamaEmbeddings(model=config.OLLAMA_EMBEDDING_MODEL_NAME, base_url=config.pro6000)  # Adjust model as needed
        
        # Create vector store
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
    
    try:
        if KNOWLEDGE_VECTORSTORE_GLOBAL is None:
            return "Knowledge base is not available. Please proceed with your medical expertise."
        
        # Perform similarity search
        relevant_docs = KNOWLEDGE_VECTORSTORE_GLOBAL.similarity_search(query, k=3)
        
        if not relevant_docs:
            return "No relevant information found in knowledge base."
        
        # Format the results
        result = "Relevant information from knowledge base:\n"
        for i, doc in enumerate(relevant_docs, 1):
            result += f"\n{i}. {doc.page_content}\n"
        
        return result
        
    except Exception as e:
        print(f"Error in knowledge base tool: {str(e)}")
        return f"Error querying knowledge base: {str(e)}. Please proceed with your medical expertise."



def run_autogen_mdt_simulation(patient_json_data: dict, patient_data_summary: str) -> str:
    """
    Runs the AutoGen MDT simulation with Dr_Chen as both Chair and Facilitator with RAG tools.
    Args:
        patient_json_data (dict): The raw JSON data for the patient.
        patient_data_summary (str): The pre-generated detailed summary for the patient.
    Returns:
        str: The full transcript of the MDT simulation.
    """
    patient_id = patient_json_data.get("病歷號", f"UnknownID_{patient_json_data.get('姓名', 'Patient')}")
    patient_name = patient_json_data.get("姓名", "N/A")
    print(f"\n--- Starting AutoGen MDT Meeting Simulation (WITH TOOLS for Chair) for Patient: {patient_name} (ID: {patient_id}) ---")

    # Base LLM config without tools for specialists
    llm_config_no_tools = {**config.llm_config_autogen}
    if "tools" in llm_config_no_tools:
        del llm_config_no_tools["tools"]

    # LLM config with tools for Dr_Chen (Chair)
    llm_config_with_tools = {
        **config.llm_config_autogen,
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "query_knowledge_base_autogen_tool",
                    "description": "Query the medical knowledge base for ILD guidelines, similar cases, and treatment recommendations",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The medical query to search for in the knowledge base"
                            }
                        },
                        "required": ["query"]
                    }
                }
            }
        ]
    }

    # DEBUGGING: Try without tools first to see if tools are causing the issue
    # Uncomment this block and comment out the tools version below if needed
    """
    chair_rheumatologist = AssistantAgent(
        name="Dr_Chen_Rheumatologist_Chair",
        system_message=(
            "You are Dr. Chen, Chair Rheumatologist and MDT Facilitator.\n\n"
            "Your role:\n"
            "1. Present the case in detail\n"
            "2. Ask Dr_Wang_Radiologist for radiological interpretation\n"
            "3. Ask Dr_Lin_Pulmonologist for pulmonology interpretation\n"
            "4. Ask Dr_Lee_Cardiologist for cardiology interpretation\n"
            "5. Go through 7 clinical questions, asking specialists and providing synthesis\n"
            "6. Get final diagnoses from all specialists\n"
            "7. Provide overall summary and end with 'MDT meeting concluded. TERMINATE_MDT_DISCUSSION_NOW'\n\n"
            "Start by presenting the case now."
        ),
        llm_config=llm_config_no_tools,  # NO TOOLS for debugging
        max_consecutive_auto_reply=1,
    )
    """
    
    # Dr_Chen as both Chair and Facilitator WITH TOOLS
    chair_rheumatologist = AssistantAgent(
        name="Dr_Chen_Rheumatologist_Chair",
        system_message=(
            "You are Dr. Chen, Chair Rheumatologist and MDT Facilitator.\n\n"
            "You have access to a medical knowledge base tool that you can use when needed.\n\n"
            "Your role:\n"
            "1. Present the case in detail\n"
            "2. Ask Dr_Wang_Radiologist for radiological interpretation\n"
            "3. Ask Dr_Lin_Pulmonologist for pulmonology interpretation\n"
            "4. Ask Dr_Lee_Cardiologist for cardiology interpretation\n"
            "5. Go through 7 clinical questions, asking specialists and providing synthesis\n"
            "6. Get final diagnoses from all specialists\n"
            "7. Provide overall summary and end with 'MDT meeting concluded. TERMINATE_MDT_DISCUSSION_NOW'\n\n"
            "IMPORTANT: Do NOT speak for other doctors. Let each specialist respond themselves.\n"
            "Start by presenting the case now."
        ),
        llm_config=llm_config_with_tools,
        max_consecutive_auto_reply=5,  # Reduced from 10
        function_map={"query_knowledge_base_autogen_tool": query_knowledge_base_autogen_tool}
    )

    # Specialists WITHOUT TOOLS (same as before)
    radiologist_specialist = AssistantAgent(
        name="Dr_Wang_Radiologist",
        system_message=(
            "You are Dr. Wang, Radiologist in this MDT meeting.\n"
            "ONLY speak when Dr_Chen directly asks for YOUR opinion or interpretation.\n"
            "Provide focused, concise radiological analysis based on HRCT findings and imaging patterns.\n"
            "Focus on: UIP vs NSIP vs AIP patterns, fibrosis, ground-glass opacities, traction bronchiectasis.\n"
            "Wait for Dr_Chen to specifically address you before responding.\n"
            "Do NOT speak for other doctors or take over the meeting."
            "**CRIT Methodology**: For any assertion you make,\n"
            "   a) **Claim**: State the main claim in one sentence.\n"
            "   b) **Supports**: List 2–3 reasons from the patient data that strongly back up your claim.\n"
            "   c) **Counters**: List 1–2 possible counter-arguments or alternative interpretations.\n"
            "   d) **Score**: For each supporting reason, give a confidence score between 0.0 and 1.0.\n"
            "   Output this section clearly under headings **Claim**, **Supports**, **Counters**, **Scores**.\n"
        ),
        llm_config=llm_config_no_tools,
        max_consecutive_auto_reply=3,
    )

    pulmonologist_specialist = AssistantAgent(
        name="Dr_Lin_Pulmonologist",
        system_message=(
            "You are Dr. Lin, Pulmonologist in this MDT meeting.\n"
            "ONLY speak when Dr_Chen directly asks for YOUR opinion or interpretation.\n"
            "Provide focused, concise pulmonology analysis based on PFTs, symptoms, and clinical course.\n"
            "Focus on: ILD classification, PPF assessment, treatment recommendations, disease progression.\n"
            "Wait for Dr_Chen to specifically address you before responding.\n"
            "Do NOT speak for other doctors or take over the meeting."
            "**CRIT Methodology**: For any assertion you make,\n"
            "   a) **Claim**: State the main claim in one sentence.\n"
            "   b) **Supports**: List 2–3 reasons from the patient data that strongly back up your claim.\n"
            "   c) **Counters**: List 1–2 possible counter-arguments or alternative interpretations.\n"
            "   d) **Score**: For each supporting reason, give a confidence score between 0.0 and 1.0.\n"
            "   Output this section clearly under headings **Claim**, **Supports**, **Counters**, **Scores**.\n"
        ),
        llm_config=llm_config_no_tools,
        max_consecutive_auto_reply=3,
    )

    cardiologist_specialist = AssistantAgent(
        name="Dr_Lee_Cardiologist",
        system_message=(
            "You are Dr. Lee, Cardiologist in this MDT meeting.\n"
            "ONLY speak when Dr_Chen directly asks for YOUR opinion or interpretation.\n"
            "Provide focused, concise cardiology analysis based on available cardiac data.\n"
            "Focus on: NT-ProBNP levels, cardiac involvement, pulmonary hypertension risk.\n"
            "If cardiac data is limited, state this clearly and briefly.\n"
            "Wait for Dr_Chen to specifically address you before responding.\n"
            "Do NOT speak for other doctors or take over the meeting."
            "**CRIT Methodology**: For any assertion you make,\n"
            "   a) **Claim**: State the main claim in one sentence.\n"
            "   b) **Supports**: List 2–3 reasons from the patient data that strongly back up your claim.\n"
            "   c) **Counters**: List 1–2 possible counter-arguments or alternative interpretations.\n"
            "   d) **Score**: For each supporting reason, give a confidence score between 0.0 and 1.0.\n"
            "   Output this section clearly under headings **Claim**, **Supports**, **Counters**, **Scores**.\n"
        ),
        llm_config=llm_config_no_tools,
        max_consecutive_auto_reply=3,
    )

    groupchat = GroupChat(
        agents=[chair_rheumatologist, radiologist_specialist, pulmonologist_specialist, cardiologist_specialist],
        messages=[],
        max_round=80,  # Reduced to prevent infinite loops
        speaker_selection_method='auto',  # Use auto speaker selection instead of custom
        allow_repeat_speaker=True,  # Prevent consecutive repeats
    )
    
    manager_llm_config = {**config.llm_config_autogen, "temperature": 0.1}
    manager = GroupChatManager(
        groupchat=groupchat, 
        llm_config=manager_llm_config,
        is_termination_msg=lambda x: x.get("content", "").rstrip().endswith("TERMINATE_MDT_DISCUSSION_NOW")
    )

    # Simplified initial prompt
    initial_prompt = (
        f"Welcome to the ILD MDT meeting for patient {patient_name} (ID: {patient_id}).\n\n"
        f"Patient Summary:\n{patient_data_summary}\n\n"
        "Dr_Chen, please present this case."
    )
    
    try:
        print(f"Initiating MDT chat with Dr_Chen...")
        print(f"Initial prompt length: {len(initial_prompt)} characters")
        
        result = chair_rheumatologist.initiate_chat(
            manager,
            message=initial_prompt,
            clear_history=True
        )
        
        print(f"Chat completed. Total messages: {len(groupchat.messages)}")
        
    except Exception as e:
        print(f"Error during MDT simulation: {e}")
        print(f"Error type: {type(e)}")
        traceback.print_exc()
        return f"Error occurred during MDT simulation: {str(e)}"

    discussion_log_autogen = [f"=== AutoGen MDT Simulation for {patient_name} (ID: {patient_id}) ===\n"]
    
    for i, msg in enumerate(groupchat.messages):
        speaker = msg.get('name', 'Unknown Speaker')
        content = msg.get('content', '')
        
        log_line = f"\n--- Message {i+1}: {speaker} ---\n{content}\n"
        discussion_log_autogen.append(log_line)

    full_transcript = "".join(discussion_log_autogen)
    print(f"\n--- AutoGen MDT Simulation (WITH TOOLS) Concluded for Patient ID: {patient_id} ---")
    return full_transcript