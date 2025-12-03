"""
Main runner for the AutoGen-based MDT simulation and evaluation.

This script implements the final "simulate-and-evaluate" architecture.
It can run in two modes on a per-patient basis:
1. Evaluation Mode: If a valid Ground Truth (GT) is found in the patient file,
the AI's answers to a standard set of questions are compared against the GT.
2. Inference Mode: If no GT is found, the AI's answers are saved directly
for review, without evaluation.

This architecture supports multi-model evaluation by:
- Reading model/tool configs to generate a unique run name.
- Saving per-run metrics to a unique JSON file.
- Scanning all past reports to generate a comprehensive, comparative Excel report.
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
from mdt_autogen_package import config

# --- Helper Functions ---

def normalize_key(key_str: str) -> str:
    """Removes punctuation and whitespace to create a comparable key."""
    if not isinstance(key_str, str):
        return ""
    return re.sub(r'[^\w]', '', key_str)

def get_answer_by_normalized_key(model_answers: dict, target_key: str):
    """Finds a value in a dictionary using a normalized key."""
    normalized_target = normalize_key(target_key)
    if not normalized_target: return None
    for key, value in model_answers.items():
        if normalize_key(key) == normalized_target:
            return value
    return None

def parse_model_answer(answer_str):
    """Parses a model's string answer into a standardized True/False/None."""
    if answer_str is None or not isinstance(answer_str, str) or not answer_str.strip():
        return None
    clean_str = answer_str.strip()
    if '是' in clean_str and '不是' not in clean_str: return True
    if '否' in clean_str or '不是' in clean_str: return False
    if '有' in clean_str: return True
    lower_str = clean_str.lower()
    if 'yes' in lower_str: return True
    if 'no' in lower_str: return False
    if 'uncertain' in lower_str: return None
    return None

# --- Metric Calculation and Reporting ---

def calculate_aggregated_metrics(run_results: list):
    """Calculates all specified metrics for a single model run."""
    metrics = {}
    valid_results = [r for r in run_results if r['status'] != 'not_applicable']
    
    num_patients = len(set(res['patient_id'] for res in valid_results))
    correct_count = len([r for r in valid_results if r['status'] == 'correct'])
    incorrect_count = len([r for r in valid_results if r['status'] == 'incorrect'])
    unanswered_count = len([r for r in valid_results if 'unanswered' in r['status']])
    total_answered = correct_count + incorrect_count
    total_comparisons = total_answered + unanswered_count

    metrics['num_patients'] = num_patients
    metrics['overall_accuracy'] = (correct_count / total_answered) * 100 if total_answered > 0 else 0
    metrics['overall_unanswered_rate'] = (unanswered_count / total_comparisons) * 100 if total_comparisons > 0 else 0

    patient_scores = {}
    for res in valid_results:
        pid = res['patient_id']
        if pid not in patient_scores: patient_scores[pid] = {'correct': 0, 'total_answered': 0}
        if res['status'] in ['correct', 'incorrect']:
            patient_scores[pid]['total_answered'] += 1
            if res['status'] == 'correct': patient_scores[pid]['correct'] += 1
    per_patient_accuracies = [(ps['correct'] / ps['total_answered']) for ps in patient_scores.values() if ps['total_answered'] > 0]
    metrics['overall_sd'] = np.std(per_patient_accuracies) if len(per_patient_accuracies) > 1 else 0

    question_stats = {q: {'correct': 0, 'incorrect': 0, 'unanswered': 0, 'not_applicable': 0} for q in config.DEFAULT_EVALUATION_QUESTIONS}
    for res in run_results:
        q = res['question']
        if q in question_stats:
            question_stats[q][res['status']] += 1
            
    metrics['question_specific_match_percent'] = {}
    metrics['question_specific_na_percent'] = {}
    metrics['question_raw_counts'] = {}

    for q, stats in question_stats.items():
        total_q_answered = stats['correct'] + stats['incorrect']
        total_q_comparisons = total_q_answered + stats['unanswered']
        metrics['question_specific_match_percent'][q] = (stats['correct'] / total_q_answered) * 100 if total_q_answered > 0 else 0
        metrics['question_specific_na_percent'][q] = (stats['unanswered'] / total_q_comparisons) * 100 if total_q_comparisons > 0 else 0
        metrics['question_raw_counts'][q] = {'correct': stats['correct'], 'incorrect': stats['incorrect']}
        
    return metrics

def calculate_p_values(model1_metrics: dict, model2_metrics: dict):
    from scipy.stats import chi2_contingency
    p_values = {}
    m1_counts = model1_metrics.get('question_raw_counts', {})
    m2_counts = model2_metrics.get('question_raw_counts', {})
    all_questions = sorted(list(set(m1_counts.keys()) | set(m2_counts.keys())))

    for question in all_questions:
        stats1 = m1_counts.get(question, {'correct': 0, 'incorrect': 0})
        stats2 = m2_counts.get(question, {'correct': 0, 'incorrect': 0})
        contingency_table = [[stats1['correct'], stats1['incorrect']], [stats2['correct'], stats2['incorrect']]]
        try:
            if sum(contingency_table[0]) == 0 or sum(contingency_table[1]) == 0: raise ValueError("Not enough data.")
            _, p, _, _ = chi2_contingency(contingency_table)
            p_values[question] = p
        except ValueError:
            p_values[question] = "N/A"
    return p_values

def save_metrics_to_excel(all_model_metrics: dict, excel_filename: str):
    print(f"\n--- Saving metrics to Excel file: {excel_filename} ---")
    if not all_model_metrics: return print("No metrics to save.")

    report_data, p_values_report = {}, {}
    sorted_questions = config.DEFAULT_EVALUATION_QUESTIONS
    metric_order = ["Total Patients Evaluated", "Mean Correctness/Accuracy (%)", "Mean Unanswered Rate (%)", "Standard Deviation (Per-Patient Accuracy)"]
    for q in sorted_questions:
        metric_order.extend([f"Match % - {q}", f"NA Rate % - {q}"])

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

    if len(all_model_metrics) == 2:
        model_names = list(all_model_metrics.keys())
        p_values_report = calculate_p_values(all_model_metrics[model_names[0]], all_model_metrics[model_names[1]])
        
    try:
        with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
            df_report.to_excel(writer, sheet_name='Overall_Metrics')
            if p_values_report:
                p_value_df = pd.DataFrame.from_dict(p_values_report, orient='index', columns=['P-Value'])
                p_value_df['Significance (p < 0.05)'] = p_value_df['P-Value'].apply(lambda p: "Yes" if isinstance(p, float) and p < 0.05 else ("No" if isinstance(p, float) else "N/A"))
                p_value_df.to_excel(writer, sheet_name='P_Value_Comparison')
        print(f"Successfully saved metrics to {excel_filename}")
    except Exception as e:
        print(f"  ERROR: Failed to save Excel file. Error: {e}")

# --- Main Execution ---

def main():
    import sys
    sys.stdout.reconfigure(encoding='utf-8')

    print("--- Setting up Knowledge Base (RAG) ---")
    setup_knowledge_base_rag(
        kb_dir=os.path.join(os.path.dirname(__file__), 'mdt_autogen_package', 'knowledge_base'),
        guidelines_file_path=os.path.join(os.path.dirname(__file__), 'mdt_autogen_package', 'knowledge_base', 'guidelines.txt'),
        similar_cases_file_path=os.path.join(os.path.dirname(__file__), 'mdt_autogen_package', 'knowledge_base', 'similar_cases.txt')
    )
    
    os.makedirs(config.EVALUATION_OUTPUT_DIR, exist_ok=True)
    inference_dir = os.path.join(config.EVALUATION_OUTPUT_DIR, "inference_outputs")
    os.makedirs(inference_dir, exist_ok=True)

    model_name_safe = config.OLLAMA_MODEL_NAME.replace(":", "_").replace("/", "_")
    tools_status = "tools_on" if config.ENABLE_TOOLS else "tools_off"
    model_run_id = f"{model_name_safe}_{tools_status}"
    
    print(f"\nStarting run for model configuration: {model_run_id}")

    csv_log_file = os.path.join(config.EVALUATION_OUTPUT_DIR, f'evaluation_log_{model_run_id}.csv')
    csv_header = ['patient_id', 'question', 'ground_truth_answer', 'model_answer_raw', 'model_answer_parsed', 'comparison_result']
    with open(csv_log_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header)
    print(f"Detailed evaluation log for this run will be saved to: {csv_log_file}")

    current_run_results = []
    patient_files = get_patient_json_files(config.PATIENT_DATA_DIR)
    print(f"Found {len(patient_files)} patient files to process.\n")
    
    standard_questions_str = "\n".join([f"- {q}" for q in config.DEFAULT_EVALUATION_QUESTIONS])

    for patient_filepath in patient_files:
        print(f"--- Processing file: {os.path.basename(patient_filepath)} ---")
        patient_data = load_single_patient_data(patient_filepath)
        if not patient_data: continue
        
        patient_id = patient_data.get("病歷號", "UnknownID")
        patient_summary = generate_dynamic_patient_summary(patient_data)
        
        model_json_output_str = run_autogen_mdt_simulation(patient_data, patient_summary, standard_questions_str)

        try:
            json_match = re.search(r'\{.*\}', model_json_output_str, re.DOTALL)
            if not json_match: raise json.JSONDecodeError("No JSON object found.", model_json_output_str, 0)
            model_answers = json.loads(json_match.group())
            if not isinstance(model_answers, dict): raise TypeError("Parsed JSON is not a dict.")
        except (json.JSONDecodeError, TypeError) as e:
            print(f"  WARNING: Could not parse JSON from model for patient {patient_id}. Error: {e}")
            model_answers = None

        conclusions_gt = patient_data.get("討論事項及結論")
        is_evaluation_run = isinstance(conclusions_gt, dict) and any(conclusions_gt.values())

        if is_evaluation_run:
            print(f"  INFO: Ground Truth found. Running in 'Evaluation Mode'.")
            ground_truth_map = {q: obj.get('Yes') for q, obj in conclusions_gt.items() if isinstance(obj, dict) and 'Yes' in obj}

            for question in config.DEFAULT_EVALUATION_QUESTIONS:
                status = ''
                gt_answer = get_answer_by_normalized_key(ground_truth_map, question)
                model_answer_raw = get_answer_by_normalized_key(model_answers, question) if model_answers else None
                model_answer_parsed = parse_model_answer(model_answer_raw)

                if gt_answer is None:
                    status = 'not_applicable'
                elif model_answer_parsed is None:
                    status = 'unanswered'
                elif model_answer_parsed == gt_answer:
                    status = 'correct'
                else:
                    status = 'incorrect'

                with open(csv_log_file, 'a', newline='', encoding='utf-8') as f:
                    csv.writer(f).writerow([patient_id, question, gt_answer, model_answer_raw, model_answer_parsed, status])
                current_run_results.append({'patient_id': patient_id, 'question': question, 'status': status})
        else:
            print(f"  INFO: No valid Ground Truth found. Running in 'Inference Mode'.")
            inference_filename = os.path.join(inference_dir, f"{patient_id}_{model_run_id}_inference.json")
            try:
                output_to_save = model_answers if model_answers else {"raw_output": model_json_output_str}
                with open(inference_filename, 'w', encoding='utf-8') as f:
                    json.dump(output_to_save, f, ensure_ascii=False, indent=2)
                print(f"  INFO: Inference output saved to: {inference_filename}")
            except Exception as e:
                print(f"  ERROR: Could not save inference output. Error: {e}")

    if not current_run_results:
        print("\nNo evaluation results were generated in this run.")
    else:
        current_run_metrics = calculate_aggregated_metrics(current_run_results)
        report_filename = os.path.join(config.EVALUATION_OUTPUT_DIR, f"report_{model_run_id}.json")
        with open(report_filename, 'w', encoding='utf-8') as f:
            json.dump(current_run_metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics for this run saved to: {report_filename}")

    all_model_metrics = {}
    report_files = glob.glob(os.path.join(config.EVALUATION_OUTPUT_DIR, "report_*.json"))
    for report_file in report_files:
        try:
            model_run_id_from_file = os.path.basename(report_file).replace("report_", "").replace(".json", "")
            with open(report_file, 'r', encoding='utf-8') as f:
                all_model_metrics[model_run_id_from_file] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load report file {report_file}. Error: {e}")

    if all_model_metrics:
        excel_report_path = os.path.join(config.EVALUATION_OUTPUT_DIR, "evaluation_summary_report.xlsx")
        save_metrics_to_excel(all_model_metrics, excel_report_path)
    else:
        print("No report files found to generate Excel summary.")

if __name__ == '__main__':
    main()
