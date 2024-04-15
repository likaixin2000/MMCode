
import os
import re
import json
import random
import base64
from io import BytesIO
import tempfile

from tqdm import tqdm
import numpy as np

from PIL import Image
from openai import OpenAI, BadRequestError


def resize(image, base_width=None, base_height=None):
    # Original dimensions
    original_width, original_height = image.size

    # Calculate new dimensions
    if base_width:
        if base_width <= original_width:
            return image
        w_percent = (base_width / float(original_width))
        new_height = int((float(original_height) * float(w_percent)))
        new_size = (base_width, new_height)
    elif base_height:
        if base_height <= original_height:
            return image
        h_percent = (base_height / float(original_height))
        new_width = int((float(original_width) * float(h_percent)))
        new_size = (new_width, base_height)
    else:
        raise ValueError("Either base_width or base_height must be specified")

    # Resize the image
    resized_img = image.resize(new_size, Image.LANCZOS)
    return resized_img


def set_random_seed(seed):
    """Set random seed for reproducibility."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)


def convert_pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


class GPT4V:
    def __init__(self):
        # Get OpenAI API Key from environment variable
        api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, images, temperature=0.0):
        prompt = (
            "You are required to solve a programming problem. " 
            + "Please enclose your code inside a ```python``` block. " 
            + " Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n" \
            + prompt
        )
        
                
        # Convert all images to base64
        base64_images = [convert_pil_image_to_base64(resize(image, base_height=480)) for image in images]

        interleaved_messages = []

        # Split the prompt and interleave text and images
        segments = re.split(r'!\[image\]\(.*?\)', prompt)
        for i, segment in enumerate(segments):
            # Text
            if len(segment) > 0:
                interleaved_messages.append({"type": "text", "text": segment})
            # Image
            if i < len(base64_images):
                interleaved_messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{base64_images[i]}",
                    }
                })


        try:
            # print(interleaved_messages)
            response = self.client.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are a professional programming contester trying to solve algorithmic problems. The problems come with a description and some images, and you should write a Python solution."}
                        ],
                    },
                    {
                        "role": "user",
                        "content": interleaved_messages
                    }
                ],
                temperature=temperature,
                max_tokens=2048,
            )
            return response.choices[0].message.content
        except BadRequestError as e:
            print("OpenAI BadRequestError:", e)
            return None
        
    def extract_code(self, response):
        pattern = r"```python(.*?)```"
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response


class GPT4:
    def __init__(self):
        # Get OpenAI API Key from environment variable
        api_key = os.environ["OPENAI_API_KEY"]
        self.client = OpenAI(api_key=api_key)

    def generate(self, prompt, images, temperature=0.0):
        prompt = "You are required to solve a programming problem. Please enclose your code inside a ```python``` block. " \
            "Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n" \
                + prompt

        interleaved_messages = []

        # Split the prompt and interleave text and images
        # image_paths = re.split(r'!\[image\]\((.*?)\)', prompt)  # Not used. We replace the images by order.
        segments = re.split(r'!\[image\]\(.*?\)', prompt)
        for i, segment in enumerate(segments):
            # Text
            if len(segment) > 0:
                interleaved_messages.append({"type": "text", "text": segment})
            # Image
            if i < len(images):
                interleaved_messages.append({"type": "text", "text": f"(Image Unavailable)"})
        response = self.client.chat.completions.create(
            model="gpt-4-1106-preview",  # gpt-4-1106-preview
            messages=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": "You are a professional programming contester trying to solve algorithmic problems. The problems come with a description and some images, and you should write a Python solution."}
                    ],
                },
                {
                    "role": "user",
                    "content": interleaved_messages
                }
            ],
            temperature=temperature,
            max_tokens=2048,
        )
        return response.choices[0].message.content
    
    def extract_code(self, response):
        pattern = r"```python(.*?)```"
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response
        

class GEMINI_PRO:
    def __init__(self):
        import google.generativeai as genai
        # Get Google API Key from environment variable
        api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        self.genai_package = genai
        self.client = genai.GenerativeModel('gemini-pro')

    def generate(self, prompt, images, temperature=0.0):
        prompt = "You are required to solve a programming problem. Please enclose your code inside a ```python``` block. " \
            "Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n" \
                 + prompt

        # Split the prompt and interleave text and images
        prompt = re.sub(r'!\[image\]\(.*?\)', "(Image Unavailable.)", prompt)
        try:
            response = self.client.generate_content(prompt,
                generation_config=self.genai_package.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            print(e)
            return None

    def extract_code(self, response):
        pattern = r"```python(.*?)```"
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response


class GEMINI_PRO_VISION:
    def __init__(self):
        import google.generativeai as genai
        # Get Google API Key from environment variable
        api_key = os.environ["GOOGLE_API_KEY"]
        genai.configure(api_key=api_key)
        self.genai_package = genai
        self.client = genai.GenerativeModel('gemini-pro-vision')

    def generate(self, prompt, images, temperature=0.0):
        meta_prompt = "You are required to solve a programming problem. Please enclose your code inside a ```python``` block. " \
            "Do not write a main() function. If Call-Based format is used, return the result in an appropriate place instead of printing it.\n\n" 

        images = [resize(image, base_height=480) for image in images]

        interleaved_messages = [meta_prompt]

        # Split the prompt and interleave text and images
        segments = re.split(r'!\[image\]\(.*?\)', prompt)
        for i, segment in enumerate(segments):
            # Text
            if len(segment) > 0:
                interleaved_messages.append(segment)
            # Image
            if i < len(images):
                interleaved_messages.append(images[i])

        # Merge continuous text segments
        messages = []
        for item in interleaved_messages:
            if not messages:
                messages.append(item)
                continue
            if isinstance(item, str) and isinstance(messages[-1], str):
                messages[-1] += item
            else:
                messages.append(item)

        try:
            response = self.client.generate_content(messages, 
                generation_config=self.genai_package.types.GenerationConfig(
                    temperature=temperature
                )
            )
            return response.text
        except Exception as e:
            print(e)
            try:
                print(response.prompt_feedback)
            except Exception as err:
                pass
            return None
    
    def extract_code(self, response):
        pattern = r"```python(.*?)```"
        # Use re.DOTALL to make '.' match any character including a newline
        matches = re.findall(pattern, response, re.DOTALL)

        if matches:
            return matches[0]
        else:
            return response
