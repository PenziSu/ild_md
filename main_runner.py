print("--- main_runner.py script started ---")
"""
Main runner for the AutoGen-based MDT simulation and evaluation.

This script implements the new "simulate-and-evaluate" workflow.
It iterates through each patient file, runs the simulation with dynamic questions
extracted from the patient's file, and then immediately evaluates the result.

This new architecture supports multi-model evaluation by:
1. Reading model and tool configuration from config.py.
2. Generating a unique name for the run (e.g., 'qwen_72b_chat_tools_on').
3. Saving the results of the current run to a corresponding JSON file (e.g., 'report_qwen_72b_chat_tools_on.json').
4. Scanning all 'report_*.json' files to aggregate results from all past runs.
5. Generating a comprehensive Excel report ('evaluation_summary_report.xlsx') with overall metrics
   and a P-value comparison sheet if two reports are found.
"""
import os
import json
import csv
import re
import glob
import numpy as np
import pandas as pd

from mdt_autogen_package.patient_data_handler import (
    get_patient_json_files,
    load_single_patient_data,
    generate_dynamic_patient_summary
)
from mdt_autogen_package.mdt_simulation_tool_CRIT import (
    run_autogen_mdt_simulation,
    setup_knowledge_base_rag
)
# Import config values
from mdt_autogen_package import config

# --- Helper Functions for Parsing and Normalization ---

def normalize_key(key_str: str) -> str:
    """Removes punctuation and whitespace to create a comparable key."""
    if not isinstance(key_str, str):
        return ""
    return re.sub(r'[^\w]', '', key_str)

def get_answer_by_normalized_key(model_answers: dict, target_key: str):
    """Finds a value in a dictionary using a normalized key."""
    normalized_target = normalize_key(target_key)
    if not normalized_target:
        return None
    for key, value in model_answers.items():
        if normalize_key(key) == normalized_target:
            return value
    return None

def parse_model_answer(answer_str):
    """Parses a model's string answer into a standardized True/False/None."""
    if answer_str is None or not isinstance(answer_str, str) or not answer_str.strip():
        return None
    
    clean_str = answer_str.strip()
    
    if '是' in clean_str and '不是' not in clean_str:
        return True
    if '否' in clean_str or '不是' in clean_str:
        return False
    if '有' in clean_str: # Handle '有' as True
        return True

    lower_str = clean_str.lower()
    if 'yes' in lower_str:
        return True
    if 'no' in lower_str:
        return False
    if 'uncertain' in lower_str:
        return None
        
    return None

# --- Metric Calculation and Reporting Functions ---

def calculate_aggregated_metrics(run_results: list, questions_for_model: list):
    """Calculates all specified metrics for a single model run."""
    metrics = {}
    num_patients = len(set(res['patient_id'] for res in run_results))
    total_questions_possible = num_patients * len(questions_for_model) if questions_for_model else 0
    
    correct_count = len([r for r in run_results if r['status'] == 'correct'])
    incorrect_count = len([r for r in run_results if r['status'] == 'incorrect'])
    unanswered_count = len([r for r in run_results if 'unanswered' in r['status']])
    
    # Overall Metrics
    metrics['num_patients'] = num_patients
    metrics['overall_accuracy'] = (correct_count / (correct_count + incorrect_count)) * 100 if (correct_count + incorrect_count) > 0 else 0
    metrics['overall_unanswered_rate'] = (unanswered_count / total_questions_possible) * 100 if total_questions_possible > 0 else 0

    # Per-Patient Accuracy for Standard Deviation
    patient_scores = {}
    for res in run_results:
        pid = res['patient_id']
        if pid not in patient_scores:
            patient_scores[pid] = {'correct': 0, 'total': 0}
        patient_scores[pid]['total'] += 1
        if res['status'] == 'correct':
            patient_scores[pid]['correct'] += 1
    per_patient_accuracies = [(ps['correct'] / ps['total']) for ps in patient_scores.values() if ps['total'] > 0]
    metrics['overall_sd'] = np.std(per_patient_accuracies) if len(per_patient_accuracies) > 1 else 0

    # Question-Specific Metrics
    question_stats = {q: {'correct': 0, 'incorrect': 0, 'unanswered': 0} for q in questions_for_model}
    for res in run_results:
        q = res['question']
        if q in question_stats:
            if res['status'] == 'correct':
                question_stats[q]['correct'] += 1
            elif res['status'] == 'incorrect':
                question_stats[q]['incorrect'] += 1
            else: # unanswered or unanswered_parse_error
                question_stats[q]['unanswered'] += 1
    
    metrics['question_specific_match_percent'] = {}
    metrics['question_specific_na_percent'] = {}
    metrics['question_raw_counts'] = {} # For p-value

    for q, stats in question_stats.items():
        total_answered = stats['correct'] + stats['incorrect']
        metrics['question_specific_match_percent'][q] = (stats['correct'] / total_answered) * 100 if total_answered > 0 else 0
        metrics['question_specific_na_percent'][q] = (stats['unanswered'] / num_patients) * 100 if num_patients > 0 else 0
        metrics['question_raw_counts'][q] = stats
        
    return metrics

def calculate_p_values(model1_metrics: dict, model2_metrics: dict):
    """Calculates p-values comparing two models for each question."""
    from scipy.stats import chi2_contingency
    p_values = {}
    
    m1_counts = model1_metrics.get('question_raw_counts', {})
    m2_counts = model2_metrics.get('question_raw_counts', {})
    
    all_questions = sorted(list(set(m1_counts.keys()) | set(m2_counts.keys())))

    for question in all_questions:
        stats1 = m1_counts.get(question, {'correct': 0, 'incorrect': 0})
        stats2 = m2_counts.get(question, {'correct': 0, 'incorrect': 0})
        
        contingency_table = [
            [stats1['correct'], stats1['incorrect']],
            [stats2['correct'], stats2['incorrect']]
        ]
        
        try:
            # Do not perform test if any group has no answered questions
            if sum(contingency_table[0]) == 0 or sum(contingency_table[1]) == 0:
                 raise ValueError("One model has no answered questions for this item.")
            _, p, _, _ = chi2_contingency(contingency_table, correction=False)
            p_values[question] = p
        except ValueError:
            p_values[question] = "N/A (Not enough data)"
            
    return p_values

def save_metrics_to_excel(all_model_metrics: dict, excel_filename="evaluation_summary_report.xlsx"):
    """Saves aggregated metrics from all runs to a structured Excel file."""
    print(f"\n--- Saving metrics to Excel file: {excel_filename} ---")
    if not all_model_metrics:
        print("No metrics to save.")
        return

    # --- Prepare Overall Metrics Sheet ---
    metric_order, sorted_questions = [], []
    report_data = {}
    
    # Determine metric order from the first model
    first_model_name = list(all_model_metrics.keys())[0]
    first_model_metrics = all_model_metrics[first_model_name]
    
    metric_order = [
        "Total Patients Evaluated", "Mean Correctness/Accuracy (%)", "Mean Unanswered Rate (%)", "Standard Deviation (Per-Patient Accuracy)"
    ]
    sorted_questions = sorted(first_model_metrics.get("question_specific_match_percent", {}).keys())
    for q in sorted_questions:
        metric_order.append(f"Match % - {q}")
        metric_order.append(f"NA Rate % - {q}")

    for model_name, metrics in all_model_metrics.items():
        col_data = {
            "Total Patients Evaluated": metrics.get('num_patients', 0),
            "Mean Correctness/Accuracy (%)": f"{metrics.get('overall_accuracy', 0):.2f}",
            "Mean Unanswered Rate (%)": f"{metrics.get('overall_unanswered_rate', 0):.2f}",
            "Standard Deviation (Per-Patient Accuracy)": f"{metrics.get('overall_sd', 0):.4f}"
        }
        for q in sorted_questions:
            col_data[f"Match % - {q}"] = f"{metrics.get('question_specific_match_percent', {}).get(q, 0):.2f}"
            col_data[f"NA Rate % - {q}"] = f"{metrics.get('question_specific_na_percent', {}).get(q, 0):.2f}"
        report_data[model_name] = col_data

    df_report = pd.DataFrame(report_data).reindex(metric_order)

    # --- Prepare P-Value Sheet ---
    p_values_report = {}
    if len(all_model_metrics) == 2:
        model_names = list(all_model_metrics.keys())
        p_values_report = calculate_p_values(all_model_metrics[model_names[0]], all_model_metrics[model_names[1]])
        
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_report.to_excel(writer, sheet_name='Overall_Metrics')
            if p_values_report:
                p_value_data = {
                    "Question": list(p_values_report.keys()),
                    "P-Value": [f"{p:.4f}" if isinstance(p, float) else p for p in p_values_report.values()],
                    "Significance (p < 0.05)": ["Yes" if isinstance(p, float) and p < 0.05 else "No" if isinstance(p, float) else "N/A" for p in p_values_report.values()]
                }
                df_p_values = pd.DataFrame(p_value_data)
                df_p_values.to_excel(writer, sheet_name='P_Value_Comparison', index=False)
        print(f"Successfully saved metrics to {excel_filename}")
    except Exception as e:
        print(f"  ERROR: Failed to save Excel file. Make sure 'pandas' and 'openpyxl' are installed. Error: {e}")

# --- Main Execution ---

def main():
    """Main function to orchestrate the simulation and evaluation process."""
    # Force UTF-8 for all stdout. Must be at the top of main.
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    # --- 1. Setup and Configuration ---
    print("--- Setting up Knowledge Base (RAG) ---")
    setup_knowledge_base_rag(
        kb_dir=os.path.join(os.path.dirname(__file__), 'mdt_autogen_package', 'knowledge_base'),
        guidelines_file_path=os.path.join(os.path.dirname(__file__), 'mdt_autogen_package', 'knowledge_base', 'guidelines.txt'),
        similar_cases_file_path=os.path.join(os.path.dirname(__file__), 'mdt_autogen_package', 'knowledge_base', 'similar_cases.txt')
    )
    
    # Ensure the output directory exists
    os.makedirs(config.EVALUATION_OUTPUT_DIR, exist_ok=True)

    # Construct unique run identifier
    model_name_safe = config.OLLAMA_MODEL_NAME.replace(":", "_").replace("/", "_")
    tools_status = "tools_on" if config.ENABLE_TOOLS else "tools_off"
    model_run_id = f"{model_name_safe}_{tools_status}"
    
    print(f"\nStarting run for model configuration: {model_run_id}")

    csv_log_file = os.path.join(config.EVALUATION_OUTPUT_DIR, f'evaluation_log_{model_run_id}.csv')
    csv_header = ['patient_id', 'question', 'ground_truth_answer', 'model_answer_raw', 'model_answer_parsed', 'comparison_result']
    with open(csv_log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    print(f"Detailed comparison log for this run will be saved to: {csv_log_file}")

    # --- 2. Main Simulation and Evaluation Loop ---
    current_run_results = []
    patient_files = get_patient_json_files(config.PATIENT_DATA_DIR) # Use config.PATIENT_DATA_DIR
    print(f"Found {len(patient_files)} patient files to process.\n")

    master_question_list = [] # To store a consistent set of questions across all patients

    for patient_filepath in patient_files:
        print(f"--- Processing file: {os.path.basename(patient_filepath)} ---")
        patient_data = load_single_patient_data(patient_filepath)
        if not patient_data: continue
        
        patient_id = patient_data.get("病歷號", "UnknownID")
        conclusions_gt = patient_data.get("討論事項及結論")
        if not isinstance(conclusions_gt, dict):
            print(f"  ERROR: '討論事項及結論' section is missing or invalid for patient {patient_id}. Skipping.")
            continue

        ground_truth_map = {q: obj['Yes'] for q, obj in conclusions_gt.items() if isinstance(obj, dict) and 'Yes' in obj and 'No' in obj and obj['Yes'] != obj['No']}
        if not ground_truth_map:
            print(f"  ERROR: No valid questions found for patient {patient_id}. Skipping.")
            continue
            
        questions_for_model = list(ground_truth_map.keys())
        if not master_question_list: master_question_list = sorted(questions_for_model) # Initialize with first patient's questions
        
        dynamic_questions_str = "\n".join([f"- {q}" for q in questions_for_model])
        patient_summary = generate_dynamic_patient_summary(patient_data)
        model_json_output_str = run_autogen_mdt_simulation(patient_data, patient_summary, dynamic_questions_str)

        try:
            json_match = re.search(r'\{.*\}', model_json_output_str, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found.", model_json_output_str, 0)
            model_answers = json.loads(json_match.group())
            if not isinstance(model_answers, dict): raise TypeError("Parsed JSON is not a dict.")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  ERROR: Could not parse response for patient {patient_id}. Logging all as parse error. Error: {e}")
            for q in questions_for_model:
                current_run_results.append({'patient_id': patient_id, 'question': q, 'status': 'unanswered_parse_error'})
            continue

        for question in questions_for_model:
            gt_answer = ground_truth_map.get(question)
            model_answer_raw = get_answer_by_normalized_key(model_answers, question)
            model_answer_parsed = parse_model_answer(model_answer_raw)
            status = 'unanswered' if model_answer_parsed is None else ('correct' if model_answer_parsed == gt_answer else 'incorrect')
            
            # Write to CSV using csv.writer to handle quoting automatically
            with open(csv_log_file, 'a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow([patient_id, question, gt_answer, model_answer_raw, model_answer_parsed, status])
            current_run_results.append({'patient_id': patient_id, 'question': question, 'status': status})

    # --- 3. Save Current Run's Metrics ---
    if not current_run_results:
        print("\nNo results were generated in this run. Skipping metric calculation.")
        return
        
    current_run_metrics = calculate_aggregated_metrics(current_run_results, master_question_list)
    report_filename = os.path.join(config.EVALUATION_OUTPUT_DIR, f"report_{model_run_id}.json")
    with open(report_filename, 'w', encoding='utf-8') as f:
        json.dump(current_run_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nMetrics for this run saved to: {report_filename}")

    # --- 4. Aggregate All Reports and Save to Excel ---
    all_model_metrics = {}
    report_files = glob.glob(os.path.join(config.EVALUATION_OUTPUT_DIR, "report_*.json")) # Search in output dir
    for report_file in report_files:
        try:
            model_run_name_from_file = os.path.basename(report_file).replace("report_", "").replace(".json", "")
            with open(report_file, 'r', encoding='utf-8') as f:
                all_model_metrics[model_run_name_from_file] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load report file {report_file}. Error: {e}")

    if all_model_metrics:
        excel_report_path = os.path.join(config.EVALUATION_OUTPUT_DIR, "evaluation_summary_report.xlsx")
        save_metrics_to_excel(all_model_metrics, excel_report_path)
    else:
        print("No report files found to generate Excel summary.")

    print("\n--- Obsolete Files ---")
    print("The following files are no longer used by the new workflow and can be safely removed if desired:")
    print("- evaluation.py, validate_gts.py, output_extractor.py")
    print(f"The 'mdt_outputs' directory is also no longer used for generating reports.")

if __name__ == '__main__':
    main()