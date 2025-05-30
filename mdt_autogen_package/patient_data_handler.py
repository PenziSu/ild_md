# mdt_autogen_package/patient_data_handler.py
import json
import os
import re
from typing import List, Dict, Any, Optional, Tuple

# If config.py is in the same package directory:
# from . import config
# If config.py is one level up (e.g. project root and this is in a sub-package):
# import config # Assuming project root is in PYTHONPATH

# For now, let's define a placeholder for PATIENT_DATA_DIR if config is not set up
# In a real scenario, this would come from your config.py
DEFAULT_PATIENT_DATA_DIR = "patient_data_files" # Example default

def get_patient_json_files(directory: str) -> List[str]:
    """
    Lists all .json files in the specified directory.
    Args:
        directory (str): The path to the directory containing patient JSON files.
    Returns:
        List[str]: A list of full file paths to the JSON files.
    """
    json_files = []
    if not os.path.isdir(directory):
        print(f"Error: Directory '{directory}' not found.")
        return json_files
        
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            json_files.append(os.path.join(directory, filename))
    if not json_files:
        print(f"No .json files found in directory '{directory}'.")
    return json_files

def load_single_patient_data(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads patient data from a single JSON file.
    Args:
        filepath (str): The full path to the patient JSON file.
    Returns:
        Optional[Dict[str, Any]]: The loaded patient data as a dictionary, or None if loading fails.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            patient_data = json.load(f)
        print(f"Successfully loaded patient data from: {filepath}({patient_data['病歷號']})")
        return patient_data
    except FileNotFoundError:
        print(f"ERROR: Patient data file not found at {filepath}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from patient data file {filepath}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred loading patient data from {filepath}: {e}")
        return None

def format_value(value, indent_level=0):
    """Helper function to format values for the summary, handling nesting."""
    indent = "  " * indent_level
    if isinstance(value, dict):
        if not value: return " (empty)"
        return "\n" + "\n".join([f"{indent}  {k}: {format_value(v, indent_level + 1)}" for k, v in value.items()])
    elif isinstance(value, list):
        if not value: return " (empty list)"
        if all(isinstance(item, (str, int, float, bool)) for item in value):
            return ", ".join(map(str, value))
        else:
            # For lists of complex items, format each one.
            # The generate_dynamic_patient_summary will iterate and call this.
            return "\n" + "\n".join([f"{indent}  - {format_value(item, indent_level + 1)}" for item in value])
    elif value is None:
        return "N/A"
    else:
        return str(value)

def generate_dynamic_patient_summary(patient_data: Dict[str, Any], max_history_chars: int = 400) -> str:
    """
    Generates a dynamic and detailed patient summary from a single patient's data dictionary,
    including all historical data.
    Args:
        patient_data (Dict[str, Any]): The dictionary containing a single patient's data.
        max_history_chars (int): Maximum characters for the 'history' field summary.
    Returns:
        str: A formatted string summary of the patient's data.
    """
    if not patient_data:
        return "Error: No patient data provided for summary generation."
    summary_parts = []

    # --- Prioritized Top-Level Information ---
    patient_name = patient_data.get("姓名", "N/A")
    patient_id = patient_data.get("病歷號", "N/A")
    age = patient_data.get("Age", "N/A")
    summary_parts.append(f"Presenting Patient: {patient_name}, ID: {patient_id}, Age: {age}")

    rheum_dx = patient_data.get("風濕科診斷", "N/A")
    summary_parts.append(f"Rheumatology Diagnosis (Initial): {rheum_dx}")
    imaging_dx = patient_data.get("影像學診斷", "N/A")
    summary_parts.append(f"Imaging Diagnosis (Initial): {imaging_dx}")
    history = str(patient_data.get("history", "N/A"))
    summary_parts.append(f"Brief History: {history[:max_history_chars]}{'...' if len(history) > max_history_chars else ''}")

    # --- Clinical Core Data ---
    clinical_core = patient_data.get("Clinical Core Data", {})
    if clinical_core:
        summary_parts.append("\n--- Clinical Core Data ---")
        current_meds_list = clinical_core.get("Current medication", clinical_core.get("Immunosuppressants", []))
        if isinstance(current_meds_list, list):
            summary_parts.append(f"  Current Medications/Immunosuppressants: {', '.join(current_meds_list) if current_meds_list else 'N/A'}")
        elif isinstance(current_meds_list, str): # Handle if it's a string
             summary_parts.append(f"  Current Medications/Immunosuppressants: {current_meds_list}")

        for key, value in clinical_core.items():
            if key not in ["Current medication", "Immunosuppressants"]: # Already handled
                summary_parts.append(f"  {key.replace('_', ' ').title()}: {format_value(value, 1)}")

    # --- Laboratory Data ---
    lab_data = patient_data.get("Laboratory", {})
    if lab_data:
        summary_parts.append("\n--- Laboratory Findings ---")
        # Immunologic profile (usually a single snapshot)
        immuno_profile = lab_data.get("Immunologic profile", lab_data.get("ANA_panel", {})) # Check both common keys
        if immuno_profile:
            summary_parts.append("  Immunologic Profile (Snapshot):") # Clarify it's usually one set
            for k_immuno, v_immuno in immuno_profile.items(): # More generic iteration
                summary_parts.append(f"    {k_immuno.replace('_', ' ').title()}: {format_value(v_immuno, 2)}")
        
        # Biologic markers in ILD (single snapshot) / Biomarkers_timeseries (all entries)
        bio_markers_single = lab_data.get("Biologic markers in ILD", {})
        bio_markers_ts = lab_data.get("Biomarkers_timeseries", [])

        if bio_markers_single:
            summary_parts.append("  Biomarkers for ILD (Snapshot):")
            for k, v in bio_markers_single.items():
                if v is not None: summary_parts.append(f"    {k}: {format_value(v,2)}")
        
        if bio_markers_ts and isinstance(bio_markers_ts, list):
            summary_parts.append("  Biomarkers Timeseries (All Entries):")
            for i, entry in enumerate(bio_markers_ts):
                if isinstance(entry, dict):
                    summary_parts.append(f"    Entry {i+1} (Date: {entry.get('日期', 'Unknown')}):")
                    for k, v in entry.items():
                        if k != '日期' and v is not None: summary_parts.append(f"      {k}: {format_value(v,3)}")
                else: summary_parts.append(f"    Entry {i+1}: {format_value(entry,2)}")

    # --- ALL Pulmonary Function Tests ---
    pft_list_raw = patient_data.get("Pulmonary Function Tests", [])
    if pft_list_raw and isinstance(pft_list_raw, list):
        summary_parts.append("\n--- All Pulmonary Function Tests ---")
        for i, pft_entry in enumerate(pft_list_raw):
            if isinstance(pft_entry, dict):
                summary_parts.append(f"  PFT Entry {i+1}:")
                for k_pft, v_pft in pft_entry.items(): # Generic iteration
                    summary_parts.append(f"    {k_pft.replace('_', ' ').title()}: {format_value(v_pft,2)}")
            else: summary_parts.append(f"  PFT Entry {i+1}: {format_value(pft_entry,1)}")

    # --- Six-Minute Walk Test(s) ---
    six_mwt_data_raw = patient_data.get("Six-minute walk test")
    # Ensure six_mwt_list is always a list, even for a single dict entry
    six_mwt_list = [six_mwt_data_raw] if isinstance(six_mwt_data_raw, dict) and six_mwt_data_raw else \
                   (six_mwt_data_raw if isinstance(six_mwt_data_raw, list) else [])
    if six_mwt_list:
        summary_parts.append("\n--- Six-Minute Walk Test(s) ---")
        for i, six_mwt_data in enumerate(six_mwt_list):
            if isinstance(six_mwt_data, dict):
                summary_parts.append(f"  6MWT Entry {i+1}:")
                for k_6mwt, v_6mwt in six_mwt_data.items():
                    summary_parts.append(f"    {k_6mwt.replace('_', ' ').title()}: {format_value(v_6mwt,2)}")
            else: summary_parts.append(f"  6MWT Entry {i+1}: {format_value(six_mwt_data,1)}")

    # --- ALL Serial HRCT Data ---
    serial_hrct_data_raw = patient_data.get("Serial HRCT")
    hrct_list_to_process = []

    if isinstance(serial_hrct_data_raw, list) and serial_hrct_data_raw:
        hrct_list_to_process = serial_hrct_data_raw
    elif isinstance(serial_hrct_data_raw, dict) and serial_hrct_data_raw:
        # Handle the "new" structure where HRCT is a dict with "Diagnoses" and "Features"
        if "Diagnoses" in serial_hrct_data_raw and isinstance(serial_hrct_data_raw["Diagnoses"], list):
            features_data = serial_hrct_data_raw.get("Features", {})
            for diag_entry in serial_hrct_data_raw["Diagnoses"]:
                if isinstance(diag_entry, dict): # Ensure diag_entry is a dict
                    # Create a combined entry for processing
                    combined_entry = {"Diagnosis_Entry": diag_entry, "Associated_Features": features_data}
                    hrct_list_to_process.append(combined_entry)
        else: # It's a dict but not the expected new structure, treat it as a single record
            hrct_list_to_process = [serial_hrct_data_raw]
    
    if hrct_list_to_process:
        summary_parts.append("\n--- All Serial HRCT Findings ---")
        for i, hrct_record_item in enumerate(hrct_list_to_process):
            summary_parts.append(f"  HRCT Record {i+1}:")
            if isinstance(hrct_record_item, dict):
                # Check if it's the combined new structure
                if "Diagnosis_Entry" in hrct_record_item and "Associated_Features" in hrct_record_item:
                    diag_entry = hrct_record_item["Diagnosis_Entry"]
                    features_data = hrct_record_item["Associated_Features"]
                    summary_parts.append(f"    Date (from Diagnosis): {diag_entry.get('Date', 'N/A')}")
                    
                    impression_parts_new = []
                    patterns = [p for p in ["UIP", "NSIP", "COP", "AIP", "RB-ILD", "DIP", "LIP", "ILA"] if diag_entry.get(p) is True]
                    if patterns: impression_parts_new.append(f"Patterns: {', '.join(patterns)}")
                    if diag_entry.get("Note"): impression_parts_new.append(f"Note: {diag_entry.get('Note')}")
                    
                    if features_data:
                        dist = features_data.get("Predominant distribution", {})
                        dist_true = [k for k, v_feat in dist.items() if v_feat is True]
                        if dist_true: impression_parts_new.append(f"Distribution: {', '.join(dist_true)}")

                        find = features_data.get("Findings", {})
                        find_true = [k for k, v_feat in find.items() if v_feat is True]
                        if find_true: impression_parts_new.append(f"Key Findings: {', '.join(find_true)}")
                        
                        fib_extent = features_data.get("Fibrosis extent_percentage", features_data.get("Fibrosis extent"))
                        if fib_extent: impression_parts_new.append(f"Fibrosis Extent: {fib_extent}")
                    
                    summary_parts.append(f"    Details: {'; '.join(impression_parts_new) if impression_parts_new else 'Structured data, see details.'}")

                else: # Standard dictionary item (old list structure or single old dict)
                    for k_hrct, v_hrct in hrct_record_item.items():
                        summary_parts.append(f"    {k_hrct.replace('_',' ').title()}: {format_value(v_hrct, 2)}")
            else: # Item in list is not a dict
                summary_parts.append(f"    Record {i+1}: {format_value(hrct_record_item,2)}")

    # --- ALL Cardiac Ultrasound Data ---
    cardiac_us_raw = patient_data.get("Cardiac ultrasound")
    cardiac_us_list = [cardiac_us_raw] if isinstance(cardiac_us_raw, dict) and cardiac_us_raw else \
                      (cardiac_us_raw if isinstance(cardiac_us_raw, list) else [])
    if cardiac_us_list:
        summary_parts.append("\n--- All Cardiac Ultrasound Reports ---")
        for i, cardiac_record in enumerate(cardiac_us_list):
            if isinstance(cardiac_record, dict):
                summary_parts.append(f"  Cardiac US Report {i+1}:")
                for k_cardiac, v_cardiac in cardiac_record.items():
                    summary_parts.append(f"    {k_cardiac.replace('_', ' ').title()}: {format_value(v_cardiac,2)}")
            else: summary_parts.append(f"  Cardiac US Report {i+1}: {format_value(cardiac_record,1)}")

    # --- CPET Data (All Entries) ---
    cpet_data_raw = patient_data.get("CPET")
    cpet_list = [cpet_data_raw] if isinstance(cpet_data_raw, dict) and cpet_data_raw else \
                (cpet_data_raw if isinstance(cpet_data_raw, list) else [])
    if cpet_list:
        summary_parts.append("\n--- Cardiopulmonary Exercise Test (CPET) Data ---")
        for i, cpet_data_item in enumerate(cpet_list):
            if isinstance(cpet_data_item, dict):
                summary_parts.append(f"  CPET Entry {i+1}:")
                for key, value in cpet_data_item.items():
                    summary_parts.append(f"    {key.replace('_', ' ').title()}: {format_value(value,2)}")
            else: summary_parts.append(f"  CPET Entry {i+1}: {format_value(cpet_data_item,1)}")
            
    return "\n".join(summary_parts)


def load_and_summarize_patients(patient_data_directory: str = DEFAULT_PATIENT_DATA_DIR) -> List[Tuple[str, Dict[str, Any], str]]:
    """
    Loads all patient JSON files from a directory and generates a summary for each.
    Args:
        patient_data_directory (str): The directory containing patient JSON files.
    Returns:
        List[Tuple[str, Dict[str, Any], str]]: A list of tuples, where each tuple contains:
            - filepath (str): The path to the patient file.
            - patient_data (Dict[str, Any]): The loaded patient data.
            - summary (str): The generated summary string for that patient.
            Returns an empty list if the directory doesn't exist or no JSON files are found.
    """
    processed_patients = []
    patient_files = get_patient_json_files(patient_data_directory)

    if not patient_files:
        print(f"No patient JSON files to process in '{patient_data_directory}'.")
        return processed_patients

    for filepath in patient_files:
        print(f"\n--- Processing file: {os.path.basename(filepath)} ---")
        patient_data = load_single_patient_data(filepath)
        if patient_data:
            summary = generate_dynamic_patient_summary(patient_data)
            processed_patients.append((filepath, patient_data, summary))
            # print(f"\n--- Summary for {os.path.basename(filepath)} ---")
            # if len(summary) > 2000:
            #     print(summary[:1000] + "\n...\n[SUMMARY TRUNCATED FOR DISPLAY]\n...\n" + summary[-500:])
            # else:
            #     print(summary)
            # print("--------------------------------------")
        else:
            print(f"Skipping summary generation for {filepath} due to loading error.")
            processed_patients.append((filepath, None, "Error: Could not load patient data."))
            
    return processed_patients

if __name__ == '__main__':
    # Example usage:
    # This will process all .json files in the DEFAULT_PATIENT_DATA_DIR
    # To use a different directory, pass it as an argument:
    # all_patient_summaries = load_and_summarize_patients("/path/to/your/patient_data_folder")
    
    print(f"Attempting to process patient files from: {DEFAULT_PATIENT_DATA_DIR}")
    all_processed_patients = load_and_summarize_patients()
    
    if all_processed_patients:
        print(f"\n\nSuccessfully processed {len(all_processed_patients)} patient files.")
        for i, (filepath, data, summary) in enumerate(all_processed_patients):
            print(f"\n--- Patient {i+1}: {os.path.basename(filepath)} ---")
            if data:
                print(f"Patient ID (from data): {data.get('病歷號', 'N/A')}")
                print("First 500 chars of summary:")
                print(summary[:500] + "..." if len(summary) > 500 else summary)
            else:
                print(summary) # This will print the error message
    else:
        print("No patient data was processed.")