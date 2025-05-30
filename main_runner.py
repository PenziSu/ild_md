# main_runner.py
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Add the project root to Python's path if mdt_autogen_package is not installed
# This allows for direct execution of main_runner.py from its location
PROJECT_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_DIR)

# Now import from the package
try:
    from mdt_autogen_package import patient_data_handler
    from mdt_autogen_package import mdt_simulation_tool_CRIT
    from mdt_autogen_package import output_extractor
    from mdt_autogen_package import config # Assuming config.py is in the package
except ImportError as e:
    print(f"Error importing package modules: {e}")
    print("Please ensure 'mdt_autogen_package' is in your PYTHONPATH or run this script from its parent directory.")
    sys.exit(1)

def main():
    print(f"--- Starting MDT AutoGen Simulation Pipeline ---")
    print(f"Using Patient Data Directory: {config.PATIENT_DATA_DIR}")
    print(f"Knowledge Base Directory: {config.KNOWLEDGE_BASE_DIR}")
    print(f"Output CSVs will be saved in: {config.OUTPUT_DIR}")

    # 1. Load and Prepare Patient Data for ALL patients in the directory
    print(f"\n--- Step 1: Loading and Summarizing ALL Patient Data from '{config.PATIENT_DATA_DIR}' ---")
    all_processed_patients_info = patient_data_handler.load_and_summarize_patients(config.PATIENT_DATA_DIR)

    if not all_processed_patients_info:
        print("No patient files processed. Exiting.") # <<< IS THIS BEING HIT?
        return

    # 2. Setup RAG (Knowledge Base) - done once
    print("\n--- Step 2: Setting up Knowledge Base and RAG (once for the session) ---")
    # Assuming create_mock_knowledge_base_files is defined in mdt_simulation_tool_CRIT.py and uses config paths
    mdt_simulation_tool_CRIT.create_mock_knowledge_base_files(
        config.KNOWLEDGE_BASE_DIR,
        config.GUIDELINES_FILE,
        config.SIMILAR_CASES_FILE
    )
    # Assuming setup_knowledge_base_rag is defined in mdt_simulation_tool_CRIT.py and uses config paths
    rag_setup_success = mdt_simulation_tool_CRIT.setup_knowledge_base_rag(
        config.KNOWLEDGE_BASE_DIR,
        config.GUIDELINES_FILE,
        config.SIMILAR_CASES_FILE
    )

    # This part you confirmed is successful
    if not rag_setup_success:
        print("\nCRITICAL WARNING: Knowledge Base RAG setup FAILED.")
        # ... (handling RAG failure) ...
    else:
        print("Knowledge Base RAG setup completed successfully.") # <<< YOU SEE THIS

    # --- Loop through each patient for MDT simulation and extraction ---
    # >>> THE PROBLEM IS LIKELY BETWEEN THE LINE ABOVE AND THE START OF THE SIMULATION
    print(f"\n--- Found {len(all_processed_patients_info)} patient(s) to process. Starting loop. ---") # Add this debug print

    for filepath, patient_json_data, patient_data_summary in all_processed_patients_info:        
        print(f"\nProcessing patient from file: {filepath}") # Add this debug print

        if not patient_json_data:
            print(f"\nSkipping MDT simulation for {os.path.basename(filepath)} due to previous loading error.")
            continue

        patient_id_main_runner = patient_json_data.get("病歷號", f"UnknownID_{os.path.basename(filepath).split('.')[0]}")
        patient_name_main_runner = patient_json_data.get("姓名", "N/A")
        
        print(f"\n\n{'='*15} Processing Patient: {patient_name_main_runner} (ID: {patient_id_main_runner} | File: {os.path.basename(filepath)}) {'='*15}")
        
        # Display a snippet of the summary for the current patient being processed
        print(f"\nPatient Summary for MDT (Patient ID: {patient_id_main_runner}):")
        if len(patient_data_summary) > 3000: 
             print(f"(Full summary is very long: {len(patient_data_summary)} chars. Using for simulation. Snippet below:)")
             print(patient_data_summary[:1500] + "\n...\n[SUMMARY TRUNCATED FOR DISPLAY IN LOG]\n...\n" + patient_data_summary[-1000:])
        else:
             print(patient_data_summary)
        print("-" * 70)

        # 3. Run MDT Simulation for the current patient
        print(f"\n--- Step 3: Running AutoGen MDT Simulation for {patient_id_main_runner} ---") # <<< ARE YOU SEEING THIS?
        
        full_transcript = None # Initialize
        try:
            full_transcript = mdt_simulation_tool_CRIT.run_autogen_mdt_simulation(
                patient_json_data,    # The dict for the current patient
                patient_data_summary  # The summary string for the current patient
            )
        except Exception as e_sim:
            print(f"!!! EXCEPTION during MDT Simulation for patient {patient_id_main_runner} !!!")
            print(f"Error type: {type(e_sim)}")
            print(f"Error message: {e_sim}")
            import traceback
            traceback.print_exc()
            print(f"Skipping further processing for patient {patient_id_main_runner} due to simulation error.")
            continue # Move to the next patient

        print(f"\n--- MDT Simulation Produced Transcript for {patient_id_main_runner} ---")
        
        # Step 4: Extract structured data
        extracted_data = output_extractor.extract_discussion_data(full_transcript, patient_id_main_runner)
        
        # Step 5: Write to CSV
        if extracted_data:
            output_extractor.write_to_csv(
                extracted_data_dict=extracted_data,
                output_directory=config.OUTPUT_DIR,
                filename_prefix=config.OUTPUT_CSV_PREFIX
            )
        else:
            print("Extraction failed or returned no data. Skipping CSV writing.")

        
        print(f"\n{'='*15} Finished Processing Patient: {patient_name_main_runner} (ID: {patient_id_main_runner}) {'='*15}")

    print("\n\n--- MDT AutoGen Simulation Pipeline Finished for All Processed Patients ---")

if __name__ == "__main__":
    main()