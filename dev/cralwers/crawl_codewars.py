import re
import os
import time
import json
import requests
import tqdm
from urllib.parse import urljoin
from bs4 import BeautifulSoup


def fetch_problem_set():
    ids = []
    print("Fetching problemset...")
    for page in tqdm.tqdm(range(1, 191)):  # Loop through pages 1 to 190
        url = f"https://www.codewars.com/kata/search/python?q=&order_by=sort_date%20desc&page={page}"

        response = requests.get(url)
        soup = BeautifulSoup(response.content, "html.parser")

        # Find the divs with class 'list-item-kata'
        problem_sets = soup.find_all('div', {"class": "list-item-kata"})
        if problem_sets is None:
            continue
        # Loop through each problem set and extract the id attribute
        for problem_set in problem_sets:
            id = problem_set.get('id')  # "id" is the attribute's name
            if id:
                ids.append(id)

    return ids


def scrape_problem(problem_id):
    save_dir = os.path.join("crawled", "codewars", "problems")

    problem_url = f"https://www.codewars.com/kata/{problem_id}"
    response = requests.get(problem_url)
    response_text = str(response.content)

    # Find the problem description markdown
    def extract_data(s):
        def unescape_str(s):
            return bytes(s, "utf-8").decode("unicode_escape")

        pattern = r'data: JSON.parse\((.*)\),\W*'
        match = re.search(pattern, s)
        if not match:
            return None
        unescaped_str = unescape_str(unescape_str(match.group(1)))
        # An example:
        # "{"routes":{},"controllerName":"code_challenges","challengeName":"Javascript filter - 1","description":"While developing a website, you detect that some of the members have troubles logging in. Searching through the code you find that all logins ending with a \"\\_\" make problems. So you want to write a function that takes an array of pairs of login-names and e-mails, and outputs an array of all login-name, e-mails-pairs from the login-names that end with \"\\_\".\n\nIf you have the input-array:\n\n```\n[ [ \"foo\", \"foo@foo.com\" ], [ \"bar_\", \"bar@bar.com\" ] ]\n```\n\nit should output\n\n```\n[ [ \"bar_\", \"bar@bar.com\" ] ]\n```\n\nYou *have to* use the *filter*-method which returns each element of the array for which the *filter*-method returns true.\n\n```javascript\nhttps://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Array/filter\n```\n\n```python\nhttps://docs.python.org/3/library/functions.html#filter\n```","activeLanguage":"javascript"}"
        data = json.loads(unescaped_str.strip('"'))
        return data

    content = extract_data(response_text)
    if content is None:
        return None
        
    problem_raw = str(content["description"])
    problem_soup = BeautifulSoup(problem_raw, "html.parser")
    images = problem_soup.find_all('img')
    img_paths = []
    for idx, img in enumerate(images, start=1):
        img_url = urljoin(problem_url, img['src'])
        img_response = requests.get(img_url)

        # Create a directory to save the image
        img_dir = os.path.join(save_dir, f"cw_{problem_id}", "images")
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
    save_path = os.path.join(save_dir, f"cw_{problem_id}", "data.json")

    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data


def crawl():
    problemset_ids = fetch_problem_set()
    print("Crawling problems...")
    for problem in tqdm.tqdm(problemset_ids):
        try:
            scrape_problem(problem)
            time.sleep(5)
        except:
            print(f"Error scrawling problem {problem}")

if __name__ == "__main__":
    crawl()
    # scrape_problem("59a67e34485a4d1ccb0000ae")
    # scrape_problem("5af4855c68e6449fbf00015c")
    