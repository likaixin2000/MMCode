import os
import time
import json
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin 

import tqdm

def fetch_problemset():
    return list(range(1, 851))


def scrape_problem(index):
    save_dir = os.path.join("crawled", "project-euler", "problems")

    problem_url = f"https://projecteuler.net/problem={index}"
    response = requests.get(problem_url)
    
    # Create BeautifulSoup object and specify the parser
    soup = BeautifulSoup(response.content, "html.parser")
    
    # find the div with class 'problem_content'
    problem_soup = soup.find('div', attrs={'class': 'problem_content'})
    
    problem_raw = str(problem_soup)
    
    images = problem_soup.find_all('img')
    img_paths = []
    for idx, img in enumerate(images, start=1):
        img_url = urljoin(problem_url, img['src'])
        img_response = requests.get(img_url)

        # Create a directory to save the image
        img_dir = os.path.join(save_dir, f"ep_{index}", "images")
        os.makedirs(img_dir, exist_ok=True)

        img_file_path = os.path.join(img_dir, f"{idx}.png")

        # Save the image
        with open(img_file_path, 'wb') as img_file:
            img_file.write(img_response.content)
      
        img_paths.append(img_file_path)
    
    # Insert local paths to problem statement
    for img, path in zip(images, img_paths):
        img.replace_with(f"![image]({path})")
    
    problem_data = {
        "url": problem_url,
        "raw_problem": problem_raw,
        "problem": problem_soup.text
    }

    # Save to file
    os.makedirs("problems", exist_ok=True)
    save_path = os.path.join(save_dir, f"ep_{index}", "data.json")
    
    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data


def crawl():
    # Fetch and save the problemset
    problems = fetch_problemset()

    for problem_index in tqdm.tqdm(problems):
        try:
            scrape_problem(problem_index)
        except:
            print(f"Error scrawling problem {problem_index}")
        time.sleep(5)


if __name__ == "__main__":
     crawl()
     