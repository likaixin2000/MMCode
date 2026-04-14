import os
import re
import time
import requests
import json
import tqdm
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def read_problemset_from_file(path):
    with open(path, 'r') as f:
        data = json.load(f)
    problems = []
    for item in data:
        problems.append({
            "problem_link": item["url"],
            "problem_id": item["url"].split('/')[-1],
            "taco_id": item["taco_id"]
        })
    return problems


def scrape_problem(problem_meta):
    """
    Params
    ----------
    problem_meta: Something like {'problem_id': 'sequences', 'problem_name': '0-1 Sequences', 'problem_link': 'https://open.kattis.com/problems/sequences'}
    """
    save_dir = os.path.join("crawled", "hackerrank", "problems")

    url = problem_meta["problem_link"]
    headers = {'User-Agent': "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
    response = requests.get(url, headers=headers)

    problem_id = url.split('/')[-2]
    soup = BeautifulSoup(response.content, "html.parser")
    problem_soup = soup.find('div', class_='challenge-body-html')
    problem_name = soup.find('div', class_='ui-icon-label page-label')
    if not problem_soup:
        print(f"Error crawling {url}")
        return None
    problem_raw = str(problem_soup)

    # Find images
    images = problem_soup.find_all('img')
    img_paths = []
    for idx, img in enumerate(images, start=1):
        img_url = urljoin(url, img['src'])
        try:
            img_response = requests.get(img_url, verify=False)
        except requests.exceptions.ConnectTimeout as e:
            print(f"Error downloading image {img_url}. Reason: {e}")
            continue

        # Create a directory to save the image
        img_dir = os.path.join(save_dir, f"hr_{problem_id}", "images")
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
        "url": url,
        "problem_id": problem_meta["problem_id"],
        "problem_name": problem_name,
        "raw_problem": problem_raw,
        "problem": problem_soup.text.strip(),
        "taco_id": problem_meta["taco_id"],
    }

    # Save to file
    save_dir = os.path.join(save_dir, f"hr_{problem_id}")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data.json")
    
    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data

if __name__ == "__main__":
    problem_meta = read_problemset_from_file("crawled/hackerrank/hackerrank.json")[-400:]
    for problem in tqdm.tqdm(problem_meta):
        scrape_problem(problem)
        # try:
        #     scrape_problem(problem)
        # except Exception as e:
        #     print(f"Error crawling {problem['problem_link']}")
        #     print(e)
        time.sleep(5)