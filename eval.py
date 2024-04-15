import sys
sys.path.append('.')

import argparse
from collections import defaultdict
from testing_utils import run_test
import json, os
import multiprocessing
import numpy as np
from typing import Dict

from tqdm import tqdm

from utils import load_problems_from_folder, load_problems_from_jsonl

IMAGE_CATEGORIES = [
    "Linear Data Structure",
    "Tree",
    "Graph",
    "2D Geometry",
    "3D Geometry",
    "Chessboard",
    "Map",
    "Patterns",
    "Math",
    "Table",
    "Pseudocode",
    "Others",
]

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
        sample = samples[task_id]
        res = []
        # loop over the generations
        for o_idx, o in enumerate(problem_generations):
            curr_res = [-2]
            try:
                curr_res = check_correctness(sample, o, timeout=20, debug=debug)
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


def compute_metrics(results, problems, k_list=[1, 10, 100], details=True):
    total = []
    correct = []

    correct_by_task = defaultdict(list)
    total_by_task = defaultdict(list)

    task_ids = []
    for task_id, res in results.items():
        all_correct = []
        all_correct_test_case = []
        for generation in res:
            verdict = np.array(generation)
            all_correct.append(np.all(verdict>0))
        task_ids.append(task_id)
        correct.append(sum(all_correct))
        total.append(len(all_correct))

        correct_by_task[task_id] = int(sum(all_correct))
        total_by_task[task_id] = int(len(all_correct))
    total = np.array(total)
    correct = np.array(correct)
    ks = k_list
    detail_pass_at_k = {f"pass@{k}": estimate_pass_at_k(total, correct, k).tolist() for k in ks if (total >= k).all()}
    result_dict = {f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()}
    if details:
        detail_metrics = {k:dict(zip(task_ids, v)) for k, v in detail_pass_at_k.items()}
        result_dict["detail"] = detail_metrics

    # Results per image category

    # WARNING: The following code only works when k=1.
    # Test accuracies
    # Image categories
    # image_type_task_acc_dict = defaultdict(lambda: [0, 0])  # Correct, Total
    # image_type_test_acc_dict = defaultdict(lambda: [0, 0])  # Correct, Total

    # for task_id in results.keys():
    #     problem = problems[task_id]
    #     image_tags = set(problem["image_tags"])
    #     for image_tag in image_tags:
    #         image_type_task_acc_dict[image_tag][0] += correct_by_task[task_id]
    #         image_type_task_acc_dict[image_tag][1] += total_by_task[task_id]
    #         image_type_test_acc_dict[image_tag][0] += result_dict["test_acc_detail"][task_id]
    #         image_type_test_acc_dict[image_tag][1] += 1
    
    # for image_tag in image_type_task_acc_dict.keys():
    #     task_stats = image_type_task_acc_dict[image_tag]
    #     task_stats.append(task_stats[0] / task_stats[1])

    # result_dict["image_category_task_acc"] = dict(image_type_task_acc_dict)

    return result_dict

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

    # Parse and return the arguments
    args = parser.parse_args()
    args.data_split = args.data_split.split(',')
    args.image_categories = args.image_categories.split(',') if args.image_categories else None
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
    # Overall metrics
    metrics = {}
    metrics["overall"] = compute_metrics(results, problems)
    metrics["categories"] = {}
    # Metrics per image category
    for image_category in IMAGE_CATEGORIES:
        image_category_problems = {k: v for k, v in problems.items() if image_category in v["image_tags"]}
        image_category_results = {k: v for k, v in results.items() if k in image_category_problems}
        image_category_metrics = compute_metrics(image_category_results, image_category_problems, details=False)
        metrics["categories"][image_category] = image_category_metrics

    json.dump(metrics, open(os.path.join(os.path.dirname(args.generation_file), f'results_{gen_file_basename}.json'), 'w'), indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)