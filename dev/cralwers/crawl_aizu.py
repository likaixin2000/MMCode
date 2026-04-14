import os
import time
import json
import requests

import tqdm
from urllib.parse import urljoin
from bs4 import BeautifulSoup

from requests.exceptions import MissingSchema

# Some apis can be found at https://judgeapi.u-aizu.ac.jp/

def fetch_problem_collections():
    root_url = 'https://judge.u-aizu.ac.jp/onlinejudge/'
    response = requests.get(root_url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    tree_div = soup.find('div', id='subtree')

    problem_collections = []
    for a_tag in tree_div.find_all('a', href=True):
        # if a_tag['href'].startswith("finder.jsp?course") or a_tag['href'].startswith("finder.jsp?volumeNo"):
        if a_tag['href'].startswith("finder.jsp?"):
            collection_url = urljoin(root_url, a_tag['href'])
            collection_name = a_tag.getText()
            collection_meta = {
                "collection_url": collection_url,
                "collection_name": collection_name
            }
            problem_collections.append(collection_meta)

    return problem_collections


# def crawl_problemset(collection_metas):
#     problems_dict = {}
#     for collection_meta in tqdm.tqdm(collection_metas):
#         problem_metas = crawl_problemset_from_collection(collection_meta)
#         for problem_meta in problem_metas:
#             problem_meta["collection_url"] = collection_meta["collection_url"]
#             problem_meta["collection_name"] = collection_meta["collection_name"]
#             # Use deduplication
#             problems_dict[problem_meta["id"]] = problem_meta
#         time.sleep(5)
#     problem_metas = list(problems_dict.values())
#     return problem_metas


# def crawl_problemset_from_collection(collection_meta):
#     identifier = collection_meta["collection_url"].rsplit('?', maxsplit=1)[-1]
#     if identifier.startswith("volumeNo"):
#         # Example: https://judge.u-aizu.ac.jp/onlinejudge/finder.jsp?volumeNo=0
#         volume_no = identifier.split('volumeNo=')[-1]
#         request_url = f"https://judgeapi.u-aizu.ac.jp/problems/volumes/{volume_no}"
#     elif identifier.startswith("course"):
#         # Example: https://judge.u-aizu.ac.jp/onlinejudge/finder.jsp?cource=NTL
#         course_id = identifier.split('course=')[-1]
#         request_url = f"https://judgeapi.u-aizu.ac.jp/problems/courses/{course_id}"
#     elif identifier.startswith("source"):
#         # Example: https://judge.u-aizu.ac.jp/onlinejudge/finder.jsp?source=JOI/Final
#         source = identifier.split('source=')[-1]
#         request_url = f"https://judgeapi.u-aizu.ac.jp/problems/cl/{source}"
#     else:
#         raise ValueError(f"Unexpected problem collection {identifier}")

#     response = requests.get(request_url)
#     # The root node has user-specific statistics which are useless, remove them
#     response_data = json.loads(response.content)["problems"]
#     return response_data


def crawl_problemset():
    response = requests.get("https://judgeapi.u-aizu.ac.jp/problems?page=0&size=1000000")
    problemset = json.loads(response.content)
    return problemset


def scrape_tests(problem_meta):
    save_dir = os.path.join("crawled", "aizu", "problems")
    problem_id = problem_meta["id"]
    tests = []
    for i in range(1, 10000):
        url = f"https://judgedat.u-aizu.ac.jp/testcases/{problem_id}/{i}"
        response = requests.get(url)
        response_data = json.loads(response.content)
        # {"problemId":"ALDS1_15_A","serial":1,"in":"100\n","out":"4\n"}
        if isinstance(response_data, list) and "code" in response_data[0] and response_data[0]["code"] == "RESOURCE_NOT_EXIST_ERROR":
            print(f"Problem id: {problem_id} Test #{i} does not exist. Finish crawling tests.")
            break
        tests.append({"input": response_data["in"], "output": response_data["out"]})

    save_path = os.path.join(save_dir, f"az_{problem_id}", "input_output.json")
    with open(save_path, 'w') as f:
        json.dump(tests, f, indent=4)
    return tests


def scrape_solutions(problem_meta):
    save_dir = os.path.join("crawled", "aizu", "problems")
    problem_id = problem_meta["id"]
    url = f"https://judgeapi.u-aizu.ac.jp/solutions/problems/{problem_id}/lang/Python3/rating?page=0&size=65536"
    response = requests.get(url)
    response_data = json.loads(response.content)
    # [{"judgeId":7146212,"userId":"jakenu0x5e","problemId":"2260","language":"Python3","version":"3.6.3","submissionDate":1670248693826,"cpuTime":2,"memory":5576,"codeSize":469,"server":9,"policy":"public","rating":1511.945436904377,"review":-1},
    solutions = []
    for submission in response_data[:100]:  # We only have corrent solutions
        if submission["policy"] != "public":
            continue
        sol_response = requests.get(f"https://judgeapi.u-aizu.ac.jp/reviews/{submission['judgeId']}")
        sol_data = json.loads(sol_response.content)
        solution = sol_data.pop("sourceCode")
        solutions.append({"meta": sol_data, "code": solution})

    save_folder = os.path.join(save_dir, f"az_{problem_id}", "submissions", "python")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "OK.json")
    with open(save_path, 'w') as f:
        json.dump(solutions, f, indent=4)
    return solutions



def scrape_problem(problem_meta):
    problem_id = problem_meta["id"]
    problem_title = problem_meta["name"]

    save_dir = os.path.join("crawled", "aizu", "problems", f"az_{problem_id}")
    url = f"https://judgeapi.u-aizu.ac.jp/resources/descriptions/en/{problem_id}"
    response = requests.get(url)
    response_data = json.loads(response.content)
    if isinstance(response_data, list) and response_data[0]["code"] == "RESOURCE_NOT_EXIST_ERROR":
        return
    problem_soup = BeautifulSoup(response_data["html"], 'html.parser')
    
    # Find images
    images = problem_soup.find_all('img')

    img_paths = []
    img_dir = os.path.join(save_dir, "images")
    if images:
        os.makedirs(img_dir, exist_ok=True)
        for idx, img in enumerate(images, start=1):
            try:
                # if 'src' not in img:
                #     continue
                img_response = requests.get(img['src'])

                # Create a directory to save the image

                img_file_path = os.path.join(img_dir, f"{idx}.png")

                # Save the image
                with open(img_file_path, 'wb') as img_file:
                    img_file.write(img_response.content)

                img_paths.append(img_file_path)
            except MissingSchema as e:
                print(e)
                continue

        # Insert local paths to problem statement
        for img, path in zip(images, img_paths):
            img.replace_with(f"![image]({path})")


    problem_raw = response_data.pop("html")
    problem_meta.update(response_data)
    problem_data = {
        "problem_meta": problem_meta,
        "problem_id": problem_meta["id"],
        "problem_title": problem_meta["name"],
        "url": url,
        "raw_problem": problem_raw,
        "problem": problem_soup.get_text(separator="\n").strip()
    }

    # Save to file
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "data.json")

    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data


# def scrape_problem(problem_meta):
#     problem_id = problem_meta["id"]
#     problem_title = problem_meta["name"]

#     save_dir = os.path.join("crawled", "aizu", "problems", f"az_{problem_id}")
#     url = f"https://judge.u-aizu.ac.jp/onlinejudge/description.jsp?id={problem_id}"
#     response = requests.get(url)
#     soup = BeautifulSoup(response.content, 'html.parser')

#     problem_soup = soup.find('div', class_='description')
    
#     problem_raw = str(problem_soup)
    
#     # Find images
#     images = problem_soup.find_all('img')

#     img_paths = []
#     img_dir = os.path.join(save_dir, "images")
#     if images:
#         os.makedirs(img_dir, exist_ok=True)
#         for idx, img in enumerate(images, start=1):
#             img_url = "https://judgeapi.u-aizu.ac.jp" + img['src']
#             print(img_url)
#             img_response = requests.get(img_url)

#             # Create a directory to save the image

#             img_file_path = os.path.join(img_dir, f"{idx}.png")

#             # Save the image
#             with open(img_file_path, 'wb') as img_file:
#                 img_file.write(img_response.content)

#             img_paths.append(img_file_path)

#         # Insert local paths to problem statement
#         for img, path in zip(images, img_paths):
#             img.replace_with(f"![image]({path})")

#     # Remove the source div tag at the bottom of the page
#     div_source = soup.find_all('div', class_='dat')[-1]
#     div_source.extract()

#     problem_data = {
#         "problem_meta": problem_meta,
#         "problem_id": problem_meta["id"],
#         "problem_title": problem_meta["name"],
#         "url": url,
#         "raw_problem": problem_raw,
#         "problem": problem_soup.get_text(separator="\n").strip()
#     }

#     # Save to file
#     os.makedirs(save_dir, exist_ok=True)
#     save_path = os.path.join(save_dir, "data.json")

#     with open(save_path, 'w') as f:
#         json.dump(problem_data, f, indent=4)
#     return problem_data


if __name__ == "__main__": 
    problem_set = crawl_problemset()
    for problem_meta in tqdm.tqdm(problem_set[-216:]):
        scrape_problem(problem_meta)
        time.sleep(1)
        scrape_tests(problem_meta)
        time.sleep(1)
        scrape_solutions(problem_meta)
        time.sleep(3)
        # try:
        #     scrape_problem(problem_meta)
        #     scrape_tests(problem_meta)
        #     scrape_solutions(problem_meta)
        #     time.sleep(5)
        # except Exception as e:
        #     print(f"Failed to scrape problem : {problem_meta['id']}")
        #     print(e)
    