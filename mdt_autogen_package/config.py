# mdt_autogen_package/config.py
import os

# --- LLM Configuration ---
phison_ip = '192.168.1.100:11434'
nstc_ip = '192.168.1.101:11434'
pro6000 = '172.22.135.15:11434'

OLLAMA_MODEL_NAME = "qwen3:8b"
OLLAMA_API_BASE = "http://" + pro6000 + "/v1"
OLLAMA_API_KEY = "ollama"
OLLAMA_EMBEDDING_MODEL_NAME = "nomic-embed-text:v1.5"
EXTRACTOR_OLLAMA_MODEL_NAME = "qwen3:8b" # For the Langchain-based extractor

# --- File and Directory Paths ---
# PROJECT_ROOT is the directory containing 'mdt_autogen_package' and 'main_runner.py'
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # This goes one level up from config.py

PATIENT_DATA_DIR = os.path.join(PROJECT_ROOT, "patient_data_files")

KNOWLEDGE_BASE_DIR = os.path.join(PROJECT_ROOT, "mdt_autogen_package", "knowledge_base") # KB inside package
GUIDELINES_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "guidelines.txt")
SIMILAR_CASES_FILE = os.path.join(KNOWLEDGE_BASE_DIR, "similar_cases.txt")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "mdt_outputs")
OUTPUT_CSV_PREFIX = "mdt_autogen_extracted_qwen3_8b" # Changed prefix to distinguish

# --- RAG Configuration ---
RAG_TOP_K = 3

# --- AutoGen LLM Configuration for Simulation Agents ---
config_list_ollama_sim = [ # Renamed for clarity in mdt_simulation.py
    {
        "model": OLLAMA_MODEL_NAME, # Use the general model name
        "base_url": OLLAMA_API_BASE,
        "api_key": OLLAMA_API_KEY,
        "api_type": "openai",
        "price": [0.0, 0.0]
    }
]
llm_config_autogen_sim = { # Renamed for clarity
    "config_list": config_list_ollama_sim,
    "cache_seed": 42,
    "temperature": 0.2,
}

# For direct use in mdt_simulation.py or other modules if they don't take llm_config_autogen_sim
llm_config_autogen = llm_config_autogen_sim 


# --- Knowledge Base Content (for mock creation if needed by mdt_simulation.create_mock_knowledge_base_files) ---
MOCK_GUIDELINE_CONTENT = """ """