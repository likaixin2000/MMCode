import argparse
from collections import defaultdict
import os
import re
import json

from PIL import Image
from tqdm import tqdm

from models import GPT4V, GPT4, GEMINI_PRO_VISION, GEMINI_PRO
from utils import load_problems_from_folder, load_problems_from_jsonl


def generate_prompt(problem):
    prompt = "\nQUESTION:\n"
    prompt += problem["question"]
    starter_code = problem["starter_code"] if len(problem.get("starter_code", [])) > 0 else None
    try:
        input_outout = json.loads(problem["input_output"])
        fn_name = (
            None if not input_outout.get("fn_name") else input_outout["fn_name"]
        )
    except ValueError:
        fn_name = None
        
    if (not fn_name) and (not starter_code):
        call_format = "\nPlease write your code using Standard IO, i.e. input() and print()."
        prompt += call_format
    else:
        call_format = "\Please write your code using Call-Based format."
        prompt += call_format

    if starter_code:
        prompt += "The starter code is provided as below. Please finish the code.\n" + starter_code + '\n'
            
    prompt += "\nANSWER:\n"
    return prompt


def main(args):
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    if args.model == "gpt4v":
        model = GPT4V()
    elif args.model == "gpt4":
        model = GPT4()
    elif args.model == "gemini_pro_vision":
        model = GEMINI_PRO_VISION()
    elif args.model == "gemini_pro":
        model = GEMINI_PRO()
    else:
        raise ValueError(f"Unknown model {args.model}")
    print(f"Running model {args.model}")
    
    if os.path.isdir(args.problems_root):
        problems = load_problems_from_folder(args.problems_root, data_split=args.data_split, image_categories=args.image_categories)
    elif os.path.isfile(args.problems_root):
        problems = load_problems_from_jsonl(args.problems_root, data_split=args.data_split, image_categories=args.image_categories)
    else:
        raise ValueError(f"Invalid path {args.problems_root}.")

    to_generate = {p["problem_id"]: args.n for p in problems}
    if os.path.exists(args.save_path):
        with open(args.save_path, 'r') as file:
            existing_results = [json.loads(item) for item in file.read().strip().splitlines()]
            # Filtering logic
            for result in existing_results:
                to_generate[result["task_id"]] -= 1

    for problem in tqdm(problems):
        for run_id in reversed(range(1, to_generate[problem["problem_id"]] + 1)):
            prompt = generate_prompt(problem)

            response = None
            attempts = 0
            # Try for up to 5 times to generate a response
            while response is None and attempts < 5:
                response = model.generate(prompt, problem["images"], temperature=args.temperature)
                attempts += 1
            
            if not response:
                print(f"Failed to generate for problem {problem['problem_id']}")
                break  # Break out of the run_id loop to move on to the next problem

            with open(args.save_path, 'a') as file:
                json.dump(
                    {
                        "task_id": problem["problem_id"], 
                        "run_id": run_id,
                        "prompt": prompt,
                        "output": model.extract_code(response),
                        "full_response": response
                    }, 
                    file  # No indents!
                )
                file.write('\n')  # Add a newline for separation between entries
            
            to_generate[problem["problem_id"]] -= 1
            

def parse_args():
    parser = argparse.ArgumentParser(description="Generate code for problems.")
    parser.add_argument("--model", type=str, default="gpt4v", choices=[
        "gpt4v", "gpt4", "gemini_pro_vision", "gemini_pro",
        ], help="Model to use for generation.")    
    parser.add_argument("--problems_root", type=str, default="../mmcode_dataset", help="Path to the root directory of problems.")
    parser.add_argument("--save_path", type=str, help="Path where the results will be saved.")
    parser.add_argument("--n", type=int, default=1, help="Number of generations per problem.")
    parser.add_argument("--data_split", type=str, default="test", help="Select the data split you want to use.")
    parser.add_argument("--image_categories", type=str, default=None, help="Select the image categories you want to use.")
    parser.add_argument("--temperature", type=float, default=0.001, help="The temperature used in the generation.")
    args = parser.parse_args()
    args.data_split = args.data_split.split(',')
    args.image_categories = args.image_categories.split(',') if args.image_categories else None
    return args

if __name__ == '__main__':
    main(parse_args())