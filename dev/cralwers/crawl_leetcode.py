import os
import time
import json
from urllib.parse import urljoin 
import requests
from bs4 import BeautifulSoup

import tqdm

def build_problem_graphql(title_slug):
    payload = {
    "operationName": 'questionData',
    "query":
    """
    query questionData($titleSlug: String!) {
      question(titleSlug: $titleSlug) {
        translatedTitle
        translatedContent
        content
        similarQuestions
        stats
        hints
        title
        titleSlug
        questionFrontendId
        codeSnippets {
          lang
          langSlug
          code
          __typename
        }
      }
    }
    """,
    "variables": { "titleSlug": title_slug }
  }
    return json.dumps(payload)


def fetch_problemset():
    response = requests.get('https://leetcode.com/api/problems/all/')
    if response.status_code == 200:
        data = response.json()['stat_status_pairs']
        with open("meta_leetcode.json", 'w') as f:
            f.write(json.dumps(data, indent=1))

        return data
    else:
        raise ValueError(f"Failed to retrieve problems, status code: {response.status_code}")




def scrape_problem(problem_entry):
    """
    An example of a problem entry:
    {
    'stat':
      {'question_id': 2918,
      'question__article__live': None,
      'question__article__slug': None,
      'question__article__has_video_solution': None,
      'question__title': 'is Array a Preorder of Some \u200cBinary Tree',
      'question__title_slug': 'is-array-a-preorder-of-some-binary-tree',
      'question__hide': False,
      'total_acs': 129,
      'total_submitted': 189,
      'frontend_question_id': 2764,
      'is_new_question': True},
    'status': None,
    'difficulty': {'level': 2},
    'paid_only': True,
    'is_favor': False,
    'frequency': 0,
    'progress': 0}
    """
    save_dir = os.path.join("crawled", "leetcode", "problems")

    problem_slug = problem_entry["stat"]["question__title_slug"]
    problem_id = problem_entry["stat"]["question_id"]
    problem_url = f"https://leetcode.com/problems/{problem_slug}"
    query_url = f"https://leetcode.com/graphql"

    local_identifier = f"lc_{problem_id}_{problem_slug}"
    # ===========================================================================
    csrf_token = ""  # TODO: Get CSRF token from LeetCode homepage or login session. It should have a length of 63 characters.
    # ===========================================================================
    headers = {
      'Content-Type': 'application/json',
      'Referer': f"https://leetcode.com/",
      'Cookie': f"csrftoken={csrf_token}",
      'X-Csrftoken': csrf_token

      }
    response = requests.post(query_url, data=build_problem_graphql(problem_slug), headers=headers)
    if response.status_code != 200:
        print(f"Unable to access LeetCode problem: {problem_slug}. Status code: {response.status_code}")
        return
    

    # Parse page content
    response_data = json.loads(response.text)
    problem_soup = BeautifulSoup(response_data["data"]["question"]["content"], 'html.parser')
    
    images = problem_soup.find_all('img')
    img_paths = []
    for idx, img in enumerate(images, start=1):
        img_url = urljoin(problem_url, img['src'])
        img_response = requests.get(img_url)

        # Create a directory to save the image
        img_dir = os.path.join(save_dir, local_identifier, "images")
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
        "raw_problem_data": response_data,
        "problem": problem_soup.text
    }

    # Save to file
    os.makedirs("problems", exist_ok=True)
    save_path = os.path.join(save_dir, local_identifier, "data.json")

    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data


def crawl(free_only=True):
    problemset = fetch_problemset()
    if free_only:
        problemset = list(filter(lambda x: not x["paid_only"], problemset))
    
    for problem in tqdm.tqdm(problemset):
        scrape_problem(problem)
        time.sleep(5)

crawl()