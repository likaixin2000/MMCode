import os
import json
import base64
import gzip
from tqdm import tqdm
from PIL import Image
from io import BytesIO


def convert_base64_to_pil_image(base64_str: str) -> Image:
    image_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(image_data))
    return image


def load_problems_from_folder(problems_root, *, return_dict=False, data_split=["test", "extra"], image_categories=None):
    if isinstance(data_split, str):
        data_split = [data_split]

    subdirs = [
        os.path.join(problems_root, d) for d in os.listdir(problems_root) 
            if os.path.isdir(os.path.join(problems_root, d))
    ]
    problems = []
    
    for subdir in tqdm(sorted(subdirs)):  # Maintain the same order
        data_json_path = os.path.join(subdir, 'data.json')
        images_folder = os.path.join(subdir, 'images')
        problem_id = os.path.basename(subdir)

        # Read data.json
        with open(data_json_path, 'r') as file:
            problem_data = json.load(file)

            # Check if the problem's data split matches the required filters
            is_correct_data_split = problem_data["data_split"] in data_split
            # If image categories are specified, check if there's at least one match
            # with the problem's image tags. If no image categories are specified,
            # consider this condition as met.
            has_matching_image_category = True if image_categories is None else \
                any(tag in image_categories for tag in problem_data["image_tags"])
            # If both conditions are met, add the problem to the list
            if not (is_correct_data_split and has_matching_image_category):
                continue

            # Load images from 'images' folder
            images = []
            image_files = sorted([f for f in os.listdir(images_folder) if f.endswith('.png')],
                                    key=lambda x: int(os.path.splitext(x)[0]))
            for img_file in image_files:
                img_path = os.path.join(images_folder, img_file)
                # image = Image.open(img_path).convert("RGBA")
                # new_image = Image.new("RGBA", image.size, "WHITE") # Create a white rgba background
                # new_image.paste(image, (0, 0), image)
                # new_image = new_image.convert('RGB')
                # del image
                # images.append(new_image)
                image = Image.open(img_path).convert('RGB')
                images.append(image)
            
            problem_data.update(
                {
                    'problem_id': problem_id,
                    'images': images,
                }
            )
            problems.append(problem_data)
    
    print(f"Loaded {len(problems)} problems.")

    if return_dict:
        return {p['problem_id']: p for p in problems}
    else:
        return problems
    

def load_problems_from_jsonl(problems_path, *, data_split=["test"], image_categories=None, return_dict=False):
    if isinstance(data_split, str):
        data_split = [data_split]

    problems = []
    with gzip.open(problems_path, 'rt') as f:
        for line in tqdm(f.readlines()):
            problem = json.loads(line)

            # Check if the problem's data split matches the required filters
            is_correct_data_split = problem["data_split"] in data_split
            # If image categories are specified, check if there's at least one match
            # with the problem's image tags. If no image categories are specified,
            # consider this condition as met.
            has_matching_image_category = True if image_categories is None else \
                any(tag in image_categories for tag in problem["image_tags"])
            # If both conditions are met, add the problem to the list
            if not (is_correct_data_split and has_matching_image_category):
                continue

            # Read images
            pil_images = [convert_base64_to_pil_image(image) for image in problem["images"]]
            del problem["images"]
            problem["images"] = pil_images
            problems.append(problem)

    print(f"Loaded {len(problems)} problems.")
    
    if return_dict:
        return {p['problem_id']: p for p in problems}
    else:
        return problems
    