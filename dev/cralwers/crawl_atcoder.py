import os
import json
import time
import random
import requests
import tqdm
from urllib.parse import urljoin
from bs4 import BeautifulSoup

def crawl_contests():
    contests = []
    for i in range(1, 12):
        url = f"https://atcoder.jp/contests/archive?page={i}"
        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract contest information
        contest_table = soup.find('tbody')
        rows = contest_table.find_all('tr') 

        if rows is None:
            raise ValueError("Failed to fetch contest data. Please check if you are sending requests too fast, which could lead to a ban.")

        for row in rows:
            # Find all <td> tags within the row
            columns = row.find_all('td')

            # Look at the second column
            contest_link_data = columns[1].find('a') if len(columns) > 1 else None

            # If we have contest link data:
            if contest_link_data:
                contest_link = contest_link_data.get('href')
                contest_name = contest_link_data.text
                contests.append({
                    'contest_name': contest_name,
                    'contest_link': urljoin(url, contest_link)}
                                )
    return contests

def crawl_tasks(contest_meta):
    # The contest page does not have tasks. Need to crawl the tasks page.
    contest_url = contest_meta["contest_link"]
    url = contest_url + "/tasks"
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the table with the tasks (assuming you know the table's structure)
    task_table = soup.find('table')
    rows = task_table.find_all('tr') if task_table else []

    tasks = []
    for row in rows:
        columns = row.find_all('td')

        if columns:
            # Extract task name and link (assuming 'a' tag is within the first 'td' tag)
            task_link_data = columns[1].find('a')
            task_name = task_link_data.text
            task_link = task_link_data.get('href')

            tasks.append({
                "task_name": task_name,
                "task_link": urljoin(url, task_link)
            })

    return tasks


def scrape_problem(contest_meta, tasks_meta):
    """
    Scrape the problem details from a given task URL. 

    Parameters
    ----------
    contest_meta : dict
        The dictionary providing the contest metadata details. 
        Example: {'contest_name': 'AtCoder Grand Contest 049', 'contest_link': 'https://atcoder.jp/contests/agc049'}

    tasks_meta : dict
        The dictionary providing the task metadata details. 
        Example: {'task_name': 'Erasing Vertices', 'task_link': 'https://atcoder.jp/contests/agc049/tasks/agc049_a'}

    Returns
    -------
    dict:
        Returns a dictionary containing problem data including contest_name, task_name, url, raw_problem, and problem. 

    Exceptions Raised
    -----------------
    Raises an exception if the task_div or task_content is not found or if there is an error crawling the task URL.
    """
    save_dir = "crawled"

    task_url = tasks_meta["task_link"]
    problem_id = task_url.rsplit("/", maxsplit=1)[-1]
    response = requests.get(task_url)
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the div with id "task-statement"
    task_div = soup.find('div', {'id': 'task-statement'})

    # Inside that div, find the span with class "lang-en"
    if not task_div:
        print(f"Error crawling {task_url}")
        return None
    task_content = task_div.find('span', {'class': 'lang-en'})
    if not task_content:
        # Check if there is only japanese version
        jap_content = task_div.find('span', {'class': 'lang-ja'})
        if jap_content is not None:
            print(f"Error crawling {task_url}. There is only japanese version of this task.")
        else:
            print(f"Error crawling {task_url}")
        return None
    problem_raw = str(task_div)

    # Find images
    images = task_content.find_all('img')
    img_paths = []
    for idx, img in enumerate(images, start=1):
        img_url = urljoin(task_url, img['src'])
        img_response = requests.get(img_url)

        # Create a directory to save the image
        img_dir = os.path.join(save_dir, f"ac_{problem_id}", "images")
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
        "contest_name": contest_meta["contest_name"],
        "task_name": tasks_meta["task_name"],
        "url": task_url,
        "raw_problem": problem_raw,
        "problem": task_content.get_text(separator="\n").strip()
    }

    # Save to file
    os.makedirs("problems", exist_ok=True)
    save_path = os.path.join(save_dir, f"ac_{problem_id}", "data.json")

    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data


def crawl():
    # Fetch and print the contests
    contest_list = crawl_contests()
    tasks_list = []
    print("Fetching problem set...")
    for contest_meta in tqdm.tqdm(contest_list):
        task_metas = crawl_tasks(contest_meta)
        for task_meta in task_metas:
            tasks_list.append((contest_meta, task_meta))
        time.sleep(10)
    
    print("Crawling problems...")
    for problem in tqdm.tqdm(tasks_list):
            scrape_problem(*problem)
            time.sleep(10)

if __name__ == "__main__":
    crawl()
