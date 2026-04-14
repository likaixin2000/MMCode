import os
import shutil
import json
import sys
sys.path.append('.')
from enum import Enum

class CodeforcesVerdict(Enum):
    ANY_VERDICT = "anyVerdict"
    ACCEPTED = "OK"
    REJECTED = "REJECTED"
    WRONG_ANSWER = "WRONG_ANSWER"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    HACKED = CHALLENGED = "CHALLENGED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    PRESENTATION_ERROR = "PRESENTATION_ERROR"
    IDLENESS_LIMIT_EXCEEDED = "IDLENESS_LIMIT_EXCEEDED"
    SECURITY_VIOLATED = "SECURITY_VIOLATED"
    CRASHED = "CRASHED"
    INPUT_PREPARATION_CRASHED = "INPUT_PREPARATION_CRASHED"
    SKIPPED = "SKIPPED"
    TESTING = "TESTING"
    PENDING_JUDGEMENT = "SUBMITTED"

# from crawl_codeforces import CodeforcesVerdict

def process_folders(root_folder, output_folder, filter_fns=None, language='python'):
    assert os.path.exists(root_folder), f"Root folder {root_folder} does not exist"
    subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]

    _filter_fns = [filter_interactive_problems,]
    if filter_fns and isinstance(filter_fns, list):
        _filter_fns.extend(filter_fns)
        for filter_fn in _filter_fns:
            subfolders = filter(filter_fn, subfolders)

    for subfolder in subfolders:
        subfolder_path = os.path.join(root_folder, subfolder)

        # Check files existence
        input_output_file = os.path.join(subfolder_path, 'input_output.json')
        submissions_folder = os.path.join(subfolder_path, 'submissions', language)
        ok_file = os.path.join(submissions_folder, 'OK.json')
        wrong_answer_file = os.path.join(submissions_folder, 'WRONG_ANSWER.json')
        
        if (
            os.path.exists(input_output_file) and 
            os.path.exists(submissions_folder) and 
            os.path.exists(ok_file) and 
            os.path.exists(wrong_answer_file)
        ):
            with open(input_output_file, 'r') as f:
                input_output_data = json.load(f)
            if isinstance(input_output_data, list) and len(input_output_data) > 0:
                output_subfolder = os.path.join(output_folder, subfolder)
                os.makedirs(output_subfolder, exist_ok=True)
                
                # Copy input_output.json
                shutil.copy(input_output_file, output_subfolder)
                
                # Read statement data
                with open(os.path.join(subfolder_path, 'data.json'), 'r') as data_file:
                    statement_data = json.load(data_file)
                    statement_text = (
                        statement_data['problem'] + '\n' \
                        # + statement_data['input_spec'] + '\n' \
                        # + statement_data['output_spec'] + '\n' 
                    )

                    sample_test_strs = [f"Input: \n{test['input']}\n\nOutput: \n{test['output']}" 
                                        for test in statement_data["sample_tests"]]

                    statement_text = (statement_text 
                                    #   + "Sample inputs and outputs:\n\n" 
                                    #   + '\n'.join(sample_test_strs)
                    )
                    
                    with open(os.path.join(output_subfolder, 'statement.txt'), 'w') as statement_file:
                        statement_file.write(statement_text)
                
                # Copy metadata.json
                shutil.copy(os.path.join(subfolder_path, 'data.json'), os.path.join(output_subfolder, 'metadata.json'))
                
                # Organize solutions based on verdict
                solutions_folder = os.path.join(output_subfolder, 'solutions')
                os.makedirs(solutions_folder, exist_ok=True)
                for verdict in CodeforcesVerdict:
                    verdict_file = os.path.join(submissions_folder, f'{verdict.value}.json')
                    if os.path.exists(verdict_file):
                        shutil.copy(verdict_file, solutions_folder)


def filter_no_images(problem_folder):
    images_folder_path = os.path.join(problem_folder, "images")

    if os.path.exists(images_folder_path):
        return False
    else:
        return True


def filter_interactive_problems(problem_folder):
    filter_str = "This is an interactive problem."
    with open(os.path.join(subfolder_path, 'data.json'), 'r') as data_file:
        statement_data = json.load(data_file)
        if filter_str in statement_data['problem']:
            return False
        else:
            return True


if __name__ == "__main__":
    # Usage example
    process_folders('/home/likaixin/mmcode/crawl/crawled/codeforces/problems', '/home/likaixin/mmcode/crawl/output/codeforces')
