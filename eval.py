import argparse
from collections import defaultdict
from testing_utils import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict

import sys
sys.path.append('.')

from tqdm import tqdm

from utils import load_problems_from_folder, load_problems_from_jsonl

TIMEOUT = 20

import logging
logging.basicConfig(
    filename='results/analysis/eval.log',  # Log file path
    filemode='w',        # 'w' to overwrite the log file each run, 'a' to append
    level=logging.INFO,  # Minimum log level
    format='%(levelname)s - %(message)s'  # Log message format
)


def check_correctness(sample, generation, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generation, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        print(f"global timeout")
        if debug:
            print(f"global timeout")
    return result[0]


def load_generation(input_file):
    generations = {}
    with open(input_file, 'r') as f:
        results = [json.loads(item) for item in f.read().strip().splitlines()]
        for _, res in enumerate(results):
            task_id = res['task_id']
            output = res['output']
            generations.setdefault(task_id, list()).append(output)
    return generations


def evaluate_generations(generations, samples, idx=None, debug=False):
    # assert len(generations.keys()) == len(samples)
    results = {}
    idx = 0
    for task_id, problem_generations in tqdm(generations.items()):
        logging.warning(f"<<< Task {task_id}")
        sample = samples[task_id]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(problem_generations):
            curr_res = [-2]
            try:
                curr_res = check_correctness(sample, o, timeout=TIMEOUT, debug=debug)
                print(curr_res)
                if debug:
                    print(f"\nSuccessful compilation of task {o_idx}!")
                fixed = []
                for e in curr_res:
                    if isinstance(e, np.ndarray):
                        e = e.item(0)
                    if isinstance(e, np.bool_):
                        e = bool(e)
                    fixed.append(e)
                curr_res = fixed
                if not np.all(curr_res):
                    if debug:
                        print(f"Results were not True for all test cases")
            except Exception as e:
                if debug:
                    print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
                break
            finally:
                assert isinstance(curr_res, list)
                res.append(curr_res)
        results[task_id] = res
        idx += 1
        logging.warning(f" Task {task_id} checked >>>")
    return results


def estimate_pass_at_k(num_samples, num_correct, k):
    """Estimates pass@k of each problem and returns them in an array."""

    def estimator(n: int, c: int, k: int) -> float:
        """Calculates 1 - comb(n - c, k) / comb(n, k)."""
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))
    import itertools
    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])



def compute_metrics_single(results, k_list):
    # Compute metrics for a single task
    total = []
    correct = []

    correct_by_task = defaultdict(list)
    total_by_task = defaultdict(list)
    
    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        for generation in res:
            verdicts = np.array(generation)
            all_correct.append(np.all(verdicts>0))
        task_ids.append(task_id)
        # Append #correct solutions of each task
        correct.append(sum(all_correct))
        # Append #total solutions of each task
        total.append(len(all_correct))

        correct_by_task[task_id] = int(sum(all_correct))
        total_by_task[task_id] = int(len(all_correct))
    # By task
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    result_dict = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    detail_metrics = {k:dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
    result_dict["detail"] = detail_metrics
    return result_dict

def compute_metrics(results, problems, k_list=[1, 5, 10]):
    # Compute metrics for all tasks
    all_result_dict = compute_metrics_single(results, k_list=k_list)
    
    # Find metrics for each image category
    tasks_by_image_category = defaultdict(dict)
    for task_id in results.keys():
        problem = problems[task_id]
        image_tags = set(problem["image_tags"])
        for image_tag in image_tags:
            tasks_by_image_category[image_tag][task_id] = results[task_id]
            
    categories_result_dict = {}
    for image_tag in tasks_by_image_category.keys():
        task_ids = tasks_by_image_category[image_tag]
        result_dict = compute_metrics_single({task_id: results[task_id] for task_id in task_ids}, k_list=k_list)
        categories_result_dict[image_tag] = result_dict

    final_result_dict = {
        'all': all_result_dict,
        'categories': categories_result_dict
    }
    return final_result_dict

def parse_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate generations against problems.')

    # Add the arguments
    parser.add_argument('--problems_root', 
                        type=str, 
                        required=True, 
                        help='The root directory where problems are stored.')
    parser.add_argument("--data_split", type=str, default="test", help="Select the data split you want to use.")
    parser.add_argument("--image_categories", type=str, default=None, help="Select the image categories you want to use.")

    parser.add_argument('--generation_file', 
                        type=str, 
                        required=True, 
                        help='File containing generations to be evaluated.')
    parser.add_argument('-k', type=str, default='1,5,10', help='The value of k for pass@k metric.')

    # Parse and return the arguments
    args = parser.parse_args()
    args.data_split = args.data_split.split(',')
    args.image_categories = args.image_categories.split(',') if args.image_categories else None
    args.k = [int(k) for k in args.k.split(',')]
    return args

def main(args):
    # Load parameters
    if os.path.isdir(args.problems_root):
        problems = load_problems_from_folder(args.problems_root, return_dict=True, data_split=args.data_split, image_categories=args.image_categories)
    elif os.path.isfile(args.problems_root):
        problems = load_problems_from_jsonl(args.problems_root, return_dict=True, data_split=args.data_split, image_categories=args.image_categories)
    else:
        raise ValueError(f"Invalid path {args.problems_root}.")
    
    generation_file = args.generation_file
    gen_file_basename = os.path.basename(generation_file)

    generations = load_generation(generation_file)

    results = evaluate_generations(generations, problems, debug=False)
    metrics = compute_metrics(results, problems, k_list=args.k)

    json.dump(metrics, open(os.path.join(os.path.dirname(args.generation_file), f'results_{gen_file_basename}.json'), 'w'), indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
