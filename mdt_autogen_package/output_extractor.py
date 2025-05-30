# mdt_autogen_package/output_extractor.py
import json
import os
import csv
import re
from datetime import datetime

from langchain_ollama import OllamaLLM as LangchainOllama
from langchain.prompts import PromptTemplate
# LLMChain is deprecated, but the runnable sequence below is the replacement.
# from langchain.chains import LLMChain 

# Import config from the package
from . import config

def create_extractor_runnable(): # Renamed from create_extractor_llm_chain
    """
    Creates an LangChain runnable sequence for extracting structured data from the MDT transcript.
    Uses LangChain Expression Language (LCEL).
    """
    json_example_for_llm_guidance = """
{
    "patient_id": "The patient's ID as mentioned in the transcript (patient's ID are like: 1398629D or 2402833J).",
    "rheumatologist_initial_presentation_summary": "A brief summary of the rheumatologist's (Dr. Chen's) initial case presentation.",
    "radiologist_initial_interpretation": "Summary of radiologist's (Dr. Wang's) initial interpretation of patient data.",
    "pulmonologist_initial_interpretation": "Summary of pulmonologist's (Dr. Lin's) initial interpretation of patient data.",
    "cardiologist_initial_interpretation": "Summary of cardiologist's (Dr. Lee's) initial interpretation of patient data.",
    "is_ild_chair_synthesis": "The Chair's (Dr. Chen's) synthesized answer for 'Is this ILD?', after team discussion.",
    "is_uip_chair_synthesis": "The Chair's (Dr. Chen's) synthesized answer for 'Is this UIP?', after team discussion.",
    "has_nsip_pattern_chair_synthesis": "The Chair's (Dr. Chen's) synthesized answer for 'Is NSIP pattern present?', after team discussion.",
    "ctd_ild_assessment_chair_synthesis": "The Chair's (Dr. Chen's) synthesized assessment of CTD-ILD (type and activity), after team discussion.",
    "is_ppf_chair_synthesis": "The Chair's (Dr. Chen's) synthesized answer for 'Is this PPF?', after team discussion.",
    "adjust_immunosuppression_chair_synthesis": "The Chair's (Dr. Chen's) synthesized recommendation on adjusting immunosuppression, after team discussion.",
    "recommend_antifibrotic_chair_synthesis": "The Chair's (Dr. Chen's) synthesized recommendation on anti-fibrotic use, after team discussion.",
    "final_diagnosis_rheumatology": "The final rheumatological diagnosis summary from Dr. Chen (Chair).",
    "final_diagnosis_pulmonology": "The final pulmonology diagnosis summary from Dr. Lin.",
    "final_diagnosis_radiology": "The final radiology diagnosis summary from Dr. Wang.",
    "final_diagnosis_cardiology": "The final cardiology diagnosis summary from Dr. Lee.",
    "final_diagnosis_overall_summary_chair": "Dr. Chen's (Chair's) final overall case summary and management plan."
}
"""
    # Escape curly braces for PromptTemplate if they are part of the JSON structure description
    json_format_instructions_for_langchain = json_example_for_llm_guidance.replace('{', '{{').replace('}', '}}')
    
    main_template_str = f"""
You are an expert medical information extractor.
Given the following Multi-Disciplinary Team (MDT) discussion transcript for an ILD patient,
please extract the requested information accurately and strictly adhere to the JSON format provided below.
The patient's ID should be extracted from the transcript if mentioned (e.g., in the Chair's presentation or initial prompt).
Focus on the CHAIR'S (Dr. Chen's) SYNTHESIZED answers for the specific clinical questions (Q1-Q7) after team input.
Also extract initial interpretations from each specialist and their final diagnoses.

Discussion Transcript:
--------------------
{{discussion_transcript}}
--------------------

Please provide the extracted information in the following JSON format.
Ensure the entire output is a single valid JSON object. Do not add any text before or after the JSON.
If information for a field is not clearly present or explicitly discussed in the transcript, use a null value or a short phrase like "Not explicitly discussed" or "N/A".

JSON Format to use:
{json_format_instructions_for_langchain}
"""
    prompt_template = PromptTemplate(
        input_variables=["discussion_transcript"], 
        template=main_template_str
    )
    
    langchain_ollama_base_url = config.OLLAMA_API_BASE
    if langchain_ollama_base_url.endswith("/v1"):
        langchain_ollama_base_url = langchain_ollama_base_url[:-3]
    elif langchain_ollama_base_url.endswith("/v1/"):
        langchain_ollama_base_url = langchain_ollama_base_url[:-4]

    extractor_llm_instance = LangchainOllama(
        model=config.EXTRACTOR_OLLAMA_MODEL_NAME,
        base_url=langchain_ollama_base_url,
        temperature=0.0
    )
    
    extractor_runnable = prompt_template | extractor_llm_instance
    return extractor_runnable

def extract_discussion_data(full_transcript: str, patient_id_fallback: str) -> dict:
    """
    Extracts structured data from the MDT discussion transcript using an LLM.
    Args:
        full_transcript (str): The complete transcript of the MDT discussion.
        patient_id_fallback (str): A patient ID to use if not found in the transcript.
    Returns:
        dict: A dictionary containing the extracted data, or None if extraction fails.
    """
    if not full_transcript or len(full_transcript.splitlines()) < 5:
        print("Warning: Transcript is very short or empty. Extraction might fail or be inaccurate.")

    extractor_runnable = create_extractor_runnable()
    extracted_data_dict = None
    
    try:
        extracted_json_str = extractor_runnable.invoke({"discussion_transcript": full_transcript})

        if extracted_json_str:
            cleaned_json_str = extracted_json_str.strip()
            match = re.search(r"\{.*\}", cleaned_json_str, re.DOTALL) 
            if match:
                json_payload_str = match.group(0)
            else:
                json_payload_str = cleaned_json_str
                if not (json_payload_str.startswith("{") and json_payload_str.endswith("}")):
                    print(f"ERROR: Extractor output does not appear to be JSON formatted: {json_payload_str[:200]}...")
                    pass 

            extracted_data_dict = json.loads(json_payload_str)
                        
            extracted_data_dict["patient_id"] = patient_id_fallback
            
            print(f"\nSuccessfully parsed extracted data for patient: {extracted_data_dict.get('patient_id')}")
            return extracted_data_dict
        else:
            print("Extractor LLM did not return any content.")
            return None
            
    except json.JSONDecodeError as je:
        print(f"ERROR: Could not parse JSON from extractor LLM: {je}")
        problematic_string = extracted_json_str if 'extracted_json_str' in locals() else "Unknown (extractor might not have returned a string)"
        print(f"Problematic string (first 500 chars): {str(problematic_string)[:500]}")
        return None
    except Exception as e_extract:
        print(f"An error occurred during LLM extraction or parsing: {e_extract} (Type: {type(e_extract)})")
        return None

def write_to_csv(extracted_data_dict: dict, output_directory: str, filename_prefix: str):
    """
    Writes the extracted data dictionary to a CSV file, appending if the file exists.
    A single CSV file is maintained per filename_prefix.
    """
    if not extracted_data_dict:
        print("No extracted data provided to write_to_csv.")
        return

    os.makedirs(output_directory, exist_ok=True)
    
    # Static filename based on prefix
    filename = os.path.join(output_directory, f"{filename_prefix}.csv")

    headers = [
        "patient_id", "rheumatologist_initial_presentation_summary",
        "radiologist_initial_interpretation", "pulmonologist_initial_interpretation", "cardiologist_initial_interpretation",
        "is_ild_chair_synthesis", "is_uip_chair_synthesis", "has_nsip_pattern_chair_synthesis",
        "ctd_ild_assessment_chair_synthesis", "is_ppf_chair_synthesis",
        "adjust_immunosuppression_chair_synthesis", "recommend_antifibrotic_chair_synthesis",
        "final_diagnosis_rheumatology", "final_diagnosis_pulmonology",
        "final_diagnosis_radiology", "final_diagnosis_cardiology",
        "final_diagnosis_overall_summary_chair"
    ]
    
    file_exists = os.path.isfile(filename)
    
    try:
        # Open in append mode ('a') if file exists, otherwise write mode ('w') to create it
        # newline='' is important to prevent blank rows in CSV
        with open(filename, mode='a' if file_exists else 'w', newline='', encoding='utf-8-sig') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
            
            if not file_exists:
                writer.writeheader()  # Write header only if file is new
            
            writer.writerow(extracted_data_dict)
        
        if file_exists:
            print(f"MDT extracted summary for patient {extracted_data_dict.get('patient_id','Unknown')} appended to CSV: {filename}")
        else:
            print(f"MDT extracted summary for patient {extracted_data_dict.get('patient_id','Unknown')} saved to new CSV: {filename}")
            
    except IOError as e:
        print(f"Error writing to CSV file {filename}: {e}")
    except Exception as e_csv:
        print(f"An unexpected error occurred during CSV writing: {e_csv}")