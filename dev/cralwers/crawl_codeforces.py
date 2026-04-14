import os
import re
import json
import time
import random
import logging
import functools
import traceback
from typing import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum

import tqdm
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from Crypto.Cipher import AES

from utils import sleep_after_execution

logger = logging.getLogger("CFCrawl")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Global variables
CSRF_TOKEN = ''

SKIP_LIST = [
    "1250_A",  # ICPC mirror, no access
    "1250_B",
    "1250_C",
    "1250_D",
    "1250_E",
    "1250_F",
    "1250_G",
    "1250_H",
    "1250_I",
    "1250_J",
    "1250_K",
    "1250_L",
    "1250_M",
    "1250_N",
]
DEFAULT_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
    'Accept-Encoding': 'gzip, deflate, br',
    'Accept-Language': 'en-US,en;q=0.9,zh-CN;q=0.8,zh;q=0.7',
    'Cache-Control': 'max-age=0',
    'Cookie': '...',  # TODO: The cookie is required to access the submission page. It can be obtained by logging in to Codeforces and checking the request headers.
    'Sec-Ch-Ua': '"Not.A/Brand";v="8", "Chromium";v="114", "Google Chrome";v="114"',
    'Sec-Ch-Ua-Mobile': '?0',
    'Sec-Ch-Ua-Platform': '"Windows"',
    'Sec-Fetch-Dest': 'document',
    'Sec-Fetch-Mode': 'navigate',
    'Sec-Fetch-Site': 'cross-site',
    'Sec-Fetch-User': '?1',
    'Upgrade-Insecure-Requests': '1'
}

with open("user_agents.txt", 'r') as f:
    USER_AGENT_LIST = [item.strip() for item in f.read().split('\n')]


def get_user_agent():
    return USER_AGENT_LIST[random.randint(0, len(USER_AGENT_LIST) - 1)]


class CodeforcesVerdict():
    ANY_VERDICT = "anyVerdict"
    ACCEPTED = "OK"
    REJECTED = "REJECTED"
    WRONG_ANSWER = "WRONG_ANSWER"
    RUNTIME_ERROR = "RUNTIME_ERROR"
    TIME_LIMIT_EXCEEDED = "TIME_LIMIT_EXCEEDED"
    MEMORY_LIMIT_EXCEEDED = "MEMORY_LIMIT_EXCEEDED"
    COMPILATION_ERROR = "COMPILATION_ERROR"
    HACKED = CHALLENGED = "CHALLENGED"
    FAILED = "FAILED"
    PARTIAL = "PARTIAL"
    PRESENTATION_ERROR = "PRESENTATION_ERROR"
    IDLENESS_LIMIT_EXCEEDED = "IDLENESS_LIMIT_EXCEEDED"
    SECURITY_VIOLATED = "SECURITY_VIOLATED"
    CRASHED = "CRASHED"
    INPUT_PREPARATION_CRASHED = "INPUT_PREPARATION_CRASHED"
    SKIPPED = "SKIPPED"
    TESTING = "TESTING"
    PENDING_JUDGEMENT = "SUBMITTED"


class CodeforcesLanguage():
    ANY_LANGUAGE = "anyProgramTypeForInvoker"
    GNU_C11 = "c.gcc11"
    CLANG_PP_20_DIAGNOSTICS = "cpp.clang++-c++20-diagnose"
    CLANG_PP_17_DIAGNOSTICS = "cpp.clang++-diagnose"
    GNU_CPP_14 = "cpp.g++14"
    GNU_CPP_17 = "cpp.g++17"
    GNU_CPP_20_64 = "cpp.gcc11-64-winlibs-g++20"
    MS_CPP_2017 = "cpp.ms2017"
    GNU_CPP_17_64 = "cpp.msys2-mingw64-9-g++17"
    CSHARP_8 = "csharp.dotnet-core"
    CSHARP_10 = "csharp.dotnet-sdk-6"
    MONO_CSHARP = "csharp.mono"
    D = "d"
    GO = "go"
    HASKELL = "haskell.ghc"
    JAVA_11 = "java11"
    JAVA_17 = "java17"
    JAVA_8 = "java8"
    KOTLIN_16 = "kotlin16"
    KOTLIN_17 = "kotlin17"
    OCAML = "ocaml"
    DELPHI = "pas.dpr"
    FPC = "pas.fpc"
    PASCALABC_NET = "pas.pascalabc"
    PERL = "perl.5"
    PHP = "php.5"
    PYTHON_2 = "python.2"
    PYTHON_3 = "python.3"
    PYPY_2 = "python.pypy2"
    PYPY_3 = "python.pypy3"
    PYPY_3_64 = "python.pypy3-64"
    RUBY_3 = "ruby.3"
    RUST_2021 = "rust.2021"
    SCALA = "scala"
    JAVASCRIPT = "v8.3"
    NODE_JS = "v8.nodejs"


class CodeforcesRetry(ValueError):
    pass


def auto_retry(max_attempts, wait_time=1):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    result = func(*args, **kwargs)
                    return result
                except CodeforcesRetry as e:
                    attempts += 1
                    logger.warning(f"Attempt {attempts}/{max_attempts} failed: {e}")
                    time.sleep(wait_time)
            raise RuntimeError(f"Maximum number of attempts ({max_attempts}) exceeded.")

        return wrapper

    return decorator


# This code is modified based on https://github.com/jeffhcs/adhoc_cf_rcpc_token_decoder/blob/master/source/cf_rcpc_token_decode.py.
# Thanks prophet!
# Codeforces Ad Hoc RCPC Token Decoder
# Author: prophet
# Date: 2020/07/14

# def decode_rcpc(redirect_page_content):
#     # parse the cipher from codeforces raw response
#     def parse_cipher(raw_response):
#         reg = "c=toNumbers\(.*?\)"
#         match = re.findall(reg, raw_response)[0]
#         match = match.replace("c=toNumbers(\"", "")
#         match = match.replace("\")", "")
#         return match

#     # convert hex string to byte array
#     def hex_to_bytes(hex_in):
#         return [int(hex_in[i * 2:i * 2 + 2], 16) for i in range(16)]

#     # decode cipher array using slow aes
#     def cipher_decode(raw_cipher):
#         aes = AES.AESModeOfOperation()
#         key = [233,238,75,3,193,208,130,41,135,24,93,39,188,162,51,120]
#         iv = [24,143,175,219,224,248,126,240,252,40,16,213,179,227,71,5]
#         mode = 2
#         orig_len = 16
#         decoded = aes.decrypt(raw_cipher, orig_len, mode, key, aes.aes.keySize["SIZE_128"], iv)
#         return decoded

#     # convert decoded token into hex
#     def bytes_to_hex(byte_array):
#         return "".join([hex(byte)[2:].zfill(2) for byte in byte_array])

#     try:
#         logger.info("Entering rcpc decoding.")
#         print("-"*100)
#         print(redirect_page_content)
#         print("-"*100)
#         raw_cipher = parse_cipher(redirect_page_content.decode('utf-8'))

#         print(f"Cipher: {raw_cipher}")

#         cipher_array = hex_to_bytes(raw_cipher)
#         decoded = cipher_decode(cipher_array)
#         token = bytes_to_hex(decoded)

#         print(f"Token : {token}")
#         return token
#     except Exception as e:
#         print("Something failed! Unable to get token.", e)
#         traceback.print_exc()


def fetch_problemset():
    problemset_url = "https://codeforces.com/api/problemset.problems"
    response = requests.get(problemset_url)
    data = response.json()

    with open("meta_codeforces.json", 'w') as f:
        f.write(json.dumps(data, indent=1))

    problems = []

    # Check if received 'OK' status
    if data['status'] == 'OK':
        for problem in data['result']['problems']:
            problems.append({
                'contest_id': problem['contestId'],
                'problem_index': problem['index'],
                'name': problem['name'],
                'type': problem['type']
            })
    else:
        raise ValueError(f"Error fetching problemset: {data['comment']}")

    return problems


# Function to scrape a problem
@auto_retry(max_attempts=10, wait_time=3)
@sleep_after_execution(5)
def scrape_problem(contest_id, problem_index):
    save_dir = os.path.join("crawled", "codeforces", "problems")
    # Scraping metadata
    problem_url = f"https://codeforces.com/problemset/problem/{contest_id}/{problem_index}"
    response = requests.get(problem_url)
    soup = BeautifulSoup(response.content, 'html.parser', from_encoding=response.encoding)

    problem_data = {}

    problem_soup = soup.find('div', {'class': 'problem-statement'})
    problem_data['problem_raw'] = str(problem_soup)

    problem_data['url'] = problem_url
    problem_data['contest_id'] = contest_id
    problem_data['problem_index'] = problem_index

    problem_data['title'] = problem_soup.find('div', {'class': 'title'}).text
    problem_data['time_limit'] = problem_soup.find('div', {'class': 'time-limit'}).find('div', {
        'class': 'property-title'}).next_sibling.text
    problem_data['memory_limit'] = problem_soup.find('div', {'class': 'memory-limit'}).find('div', {
        'class': 'property-title'}).next_sibling.text

    # Finding and Saving Images in the "problem_statement" div
    images = problem_soup.find_all('img')
    img_paths = []
    for idx, img in enumerate(images, start=1):
        img_url = urljoin(problem_url, img['src'])
        img_response = requests.get(img_url)

        # Create a directory to save the image
        img_dir = os.path.join(save_dir, f"cf_{contest_id}_{problem_index}", "images")
        os.makedirs(img_dir, exist_ok=True)

        img_file_path = os.path.join(img_dir, f"{idx}.png")

        # Save the image
        with open(img_file_path, 'wb') as img_file:
            img_file.write(img_response.content)

        img_paths.append(img_file_path)

    # # Insert local paths to problem statement
    # for img, path in zip(images, img_paths):
    #     img.replace_with(f"![image]({path})")
    for i, img in enumerate(images):
        img.replace_with(f"![image]({i + 1}.png)")

    problem_data['problem'] = problem_soup.find('div', {'class': 'header'}).next_sibling.text.strip()

    # Gather input paragraphs
    input_spec = problem_soup.find('div', {'class': 'input-specification'})
    if input_spec is not None:
        input_paragraphs = input_spec.find('div', {'class': 'section-title'}).find_next_siblings()
        problem_data['input_spec'] = "\n".join(p.text.strip() for p in input_paragraphs)
    else:
        problem_data['input_spec'] = ''

    # Gather output paragraphs
    output_spec = problem_soup.find('div', {'class': 'output-specification'})
    if output_spec is not None:
        output_paragraphs = output_spec.find('div', {'class': 'section-title'}).find_next_siblings()
        problem_data['output_spec'] = "\n".join(p.text.strip() for p in output_paragraphs)
    else:
        problem_data['output_spec'] = ''

    # Gather note paragraphs
    note = problem_soup.find('div', {'class': 'note'})
    if note is not None:
        note_paragraphs = note.find('div', {'class': 'section-title'}).find_next_siblings()
        problem_data['note'] = "\n".join(p.text.strip() for p in note_paragraphs)
    else:
        problem_data['note'] = ""

    # Find the sample tests:
    sample_tests = problem_soup.find('div', {'class': 'sample-tests'})

    test_input_blocks = sample_tests.find_all('div', {'class': 'input'})
    test_output_blocks = sample_tests.find_all('div', {'class': 'output'})

    sample_tests_data = []

    for input_block, output_block in zip(test_input_blocks, test_output_blocks):
        sample_test = {}
        sample_test['input'] = input_block.find('pre').get_text(separator="\n").strip()
        sample_test['output'] = output_block.find('pre').get_text(separator="\n").strip()
        sample_tests_data.append(sample_test)

    # Add sample tests to the problem data dictionary
    problem_data['sample_tests'] = sample_tests_data

    # Save to file
    save_path = os.path.join(save_dir, f"cf_{contest_id}_{problem_index}", "data.json")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, 'w') as f:
        json.dump(problem_data, f, indent=4)
    return problem_data


@auto_retry(max_attempts=10, wait_time=5)
@sleep_after_execution(5)
def _set_filter(contest_id, problem_index, verdict_name, language):
    url = f"https://codeforces.com/problemset/status/{contest_id}/problem/{problem_index}"
    session = requests.Session()
    session.headers["User-Agent"] = get_user_agent()
    response = session.get(url)
    # Try to find the csrf token. Cookies are automatically handled by Session object.
    soup = BeautifulSoup(response.text, 'html.parser')
    csrf_token_tag = soup.find("meta", {'name': "X-Csrf-Token"})
    if csrf_token_tag:
        csrf_token = csrf_token_tag['content']
    else:
        print("Can not find csrf token for url:", url)
        raise CodeforcesRetry("Can not find csrf token!")
    # This is actually not required by the Codeforces server.
    # Just needed a place to store the token.
    session.headers["X-Csrf-Token"] = csrf_token
    # TODO: Fix this ugly patch
    global CSRF_TOKEN
    CSRF_TOKEN = csrf_token

    # Set filter
    data = {
        'csrf_token': csrf_token,
        'action': "setupSubmissionFilter",
        'frameProblemIndex': problem_index,
        'verdictName': verdict_name,
        'programTypeForInvoker': language,  # python.3
        'comparisonType': "NOT_USED",
        'judgedTestCount': "",
        'participantSubstring': "",
    }
    response = session.post(url, data=data, headers=session.headers, allow_redirects=False)
    if response.status_code != 302:
        raise CodeforcesRetry(f"Wrong response status code: {response.status_code}. Filters are not set.")
    # The response status code should be 302, which asks for a redirect.
    # We manually GET that page to retrieve the csrf token.
    response = session.get(response.headers['Location'])
    return session


def _reset_filter(session, contest_id, problem_index):
    url = f"https://codeforces.com/problemset/status/{contest_id}/problem/{problem_index}?order=BY_ARRIVED_DESC"
    data = {
        'csrf_token': session.headers["X-Csrf-Token"],
        'action': "resetStatusFilter",
    }
    response = session.post(url, data=data, allow_redirects=False)
    if response.status_code != 302:
        raise ValueError(f"Wrong response status code: {response.status_code}. Filters are not cleared.")
    return None


@contextmanager
def set_cf_filter(contest_id,
                  problem_index,
                  verdict_name,
                  language,
                  ):
    """
    A context manager to set a filter for Codeforces submissions and automatically reset it after execution.
    To filter the submissions, the user has to post filter configs to the server, which is recorded per contest on the server side.

    Parameters:
    ----------
    contest_id : str
        The ID of the Codeforces contest.
    problem_index : str
        The index of the problem within the contest (e.g., 'A', 'B', 'C', etc.).
    verdict_name : str
        The name of the verdict to filter submissions (e.g., CodeforcesVerdict.ACCEPTED, etc.).
    language : str
        The programming language for the filter (e.g., CodeforcesLanguage.PYTHON3, etc.).


    Usage:
    -------
    Use this context manager with the `with` statement to set a Codeforces submission filter for a specific context.
    The filter will be automatically reset after execution of the code within the context.
    """
    session = None
    try:
        # Set filter
        session = _set_filter(contest_id, problem_index, verdict_name, language)
        yield session
    finally:
        # Reset filter
        if session is not None:
            _reset_filter(session, contest_id, problem_index)


def format_space(text):
    return text.replace('\xa0', ' ').strip()


def pick_diverse_elements(data: list, key: str, max_count: int):
    """
    Pick diverse elements from the data list based on the specified key, ensuring the max_count limit is met.

    Parameters:
        data (list): A list of dictionaries containing the elements to pick from.
        key (str): The key in the dictionaries to determine uniqueness.
        max_count (int): The maximum number of diverse elements to pick.

    Returns:
        list: A list of diverse elements based on the specified key and max_count.
    """

    if len(data) <= max_count:
        # If the data list is smaller than the max_count, return the sorted data list.
        return sorted(data, key=lambda item: item[key])

    unique_keys = set()
    indexed_dict = defaultdict(list)
    for item in data:
        unique_keys.add(item[key])
        indexed_dict[item[key]].append(item)

    ret = []
    while True:
        for cur_key in unique_keys:
            ret.append(indexed_dict[cur_key].pop(0))
            if len(ret) == max_count:
                # If the desired max_count of unique elements is reached, return the result.
                return ret
        # Keep remaining keys by filtering out those with empty lists.
        unique_keys = set(filter(lambda key: len(indexed_dict[key]) > 0, unique_keys))


def _crawl_test_cases_by_submission_id(acc_submission_id) -> list:
    url = 'https://codeforces.com/data/submitSource'

    headers = {
        "Cookie": (""),  # TODO: The cookie is required to access the submission page. It can be obtained by logging in to Codeforces and checking the request headers.
        "Referer": "https://codeforces.com/contest/",
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"),
    }
    payload = {'submissionId': str(acc_submission_id), 'csrf_token': "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"}  # TODO: The csrf token is required 

    response = requests.post(url, data=payload, headers=headers)

    if response.status_code != 200:
        logger.warning(f"Unable to fetch data for submission {acc_submission_id}, status code: {response.status_code}")
        return None

    data = response.json()

    # The submission may be a virtual one. No access to test cases.
    if "testCount" not in data:
        return None

    test_cases_cnt = int(data["testCount"])
    verdict_ok = [data[f"verdict#{i}"] == "OK" for i in range(1, test_cases_cnt + 1)]
    assert all(verdict_ok), "Submission results are not all OK. " \
                            "Please make sure the submission is ACCEPTED."

    result = []
    for test_idx in range(1, test_cases_cnt + 1):
        inp = data[f"input#{test_idx}"]
        ans = data[f"answer#{test_idx}"]  # The "answer" key holds the expected output
        # Skip incomplete test cases
        if inp.endswith("..."):
            continue
        result.append(
            {
                "input": inp,
                # "output": data[f"output#{test_idx}"],
                "output": ans
            }
        )
    # Note: When we have access to test cases but none is usable, the result is an empty list.
    return result


def crawl_tests(contest_id, problem_index):
    test_cases = []

    # Find an accepted submission_id
    accepted_submissions = find_submission_ids(contest_id, problem_index,
                                               verdict_name=CodeforcesVerdict.ACCEPTED,
                                               language=CodeforcesLanguage.ANY_LANGUAGE,
                                               stop_criteria=lambda x: True
                                               # We only need one submission, so stop directly
                                               )

    if len(accepted_submissions) == 0:
        logger.warning(f"No accepted submissions found for {contest_id}_{problem_index}.")
        return []

    test_cases = None
    for accepted_submission in random.choices(accepted_submissions, k=3):
        # Choose an accepted submission_id
        acc_submission_id = accepted_submission["submission_id"]

        # Crawl test cases using the accepted submission_id
        test_cases = _crawl_test_cases_by_submission_id(acc_submission_id)
        # Sometimes we do not have access to the test cases in the submission page because it may be a virtual submission
        if test_cases is None:
            # Try next submission
            continue
        else:
            break

    if test_cases is None:
        logger.warning(f"All attempts to get test cases for {contest_id}{problem_index} failed.")
        return None

    if len(test_cases) == 0:
        logger.warning(f"No test cases found for {contest_id}{problem_index}.")

    return test_cases


@auto_retry(max_attempts=10, wait_time=3)
@sleep_after_execution(15)
def crawl_submission_code(contest_id, submission_id):
    # This method often fails due to unknown reason.

    url = f"https://codeforces.com/problemset/submission/{contest_id}/{submission_id}"

    headers = {'User-Agent': get_user_agent()}
    response = requests.get(url, headers=headers, allow_redirects=False)

    soup = BeautifulSoup(response.content, 'html.parser')
    submission_code_node = soup.find('pre', id='program-source-text')

    if not submission_code_node:
        raise CodeforcesRetry(f"contest {contest_id}, submission id {submission_id}")

    submission_code = submission_code_node.text.strip()

    return submission_code


@sleep_after_execution(15)
@auto_retry(max_attempts=10, wait_time=3)
def crawl_submissions_page(session, contest_id, problem_index, page):
    # Construct the URL of the problem status page
    status_url = f"https://codeforces.com/problemset/status/{contest_id}/problem/{problem_index}/page/{page}?order=BY_ARRIVED_DESC"

    # Send an HTTP GET request to fetch the page content
    response = session.get(status_url)

    # Parse the page content using BeautifulSoup
    soup = BeautifulSoup(response.content, "html.parser")

    # Find all submission rows in the table
    submission_rows = soup.select(".status-frame-datatable tr")
    if not submission_rows:
        raise CodeforcesRetry(f"contest {contest_id}, problem_index {problem_index}, submission id {submission_id}")
    # Initialize an empty list to store all submissions on the page
    submissions = []

    # Skip the first row (header row) and iterate through each submission row
    submission_rows = submission_rows[1:]

    # If the length is zero, it means probably we are on a wrong page.
    # Note when the filter does not return any submission, the length is one. The first elements is "No items".
    if len(submission_rows) == 0:
        raise ValueError(
            f"No submissions found for problem{contest_id}_{problem_index}, page={page}. Wrong parameters?")
    if len(submission_rows) == 1 and "No items" in submission_rows[0].text:
        return [], 0

    for row in submission_rows:
        cells = row.find_all("td")

        # Extract submission information from the cells and format the space
        # TODO: Some questions belong to two different contests, and submission is only available at the contest it is submitted to.
        # For example, 77_A and 80_C are the same question.
        # https://codeforces.com/problemset/submission/77/144589707 won't work,
        # but https://codeforces.com/problemset/submission/80/144589707 will.
        # For now, we just ignore the submissions with wrong parent contests.
        submission_url = cells[0].find('a').get('href')
        submission_parent_contest = submission_url.split('/')[-2]
        if submission_parent_contest != str(contest_id):
            logging.info(f"Duplicate contests found: {contest_id}, {submission_parent_contest}. Ignoring submission.")
            continue

        submission_id = format_space(cells[0].text)
        language = format_space(cells[4].text)
        verdict = format_space(cells[5].text)
        run_time = format_space(cells[6].text)
        memory = format_space(cells[7].text)

        # Create a dictionary containing submission information
        submission_info = {
            # "submission_url": submission_url,
            "submission_id": submission_id,
            "language": language,
            "verdict": verdict,
            "run_time": run_time,
            "memory": memory
        }

        # Append the submission information to the list
        submissions.append(submission_info)

    # Find out the maximum number of pages
    pagination = soup.find_all('div', class_='pagination')[-1]  # There are multiple paginations in the page
    span_pages = pagination.find_all('span', class_='page-index')
    if span_pages:
        max_pages = int(span_pages[-1].text)
    else:
        # There is no page spans when the maximum pages is 1
        max_pages = 1
    return submissions, max_pages

@sleep_after_execution(15)
def find_submission_ids(contest_id, problem_index, verdict_name, language,
                        stop_criteria: Callable = None):
    """
    Finds submission IDs that match the specified filter criteria by iterating the submission list page of Codeforces.
    Due to the limited function of the filtering function of Codeforces,
    this function only guarantees enough submission IDs are obtatined.
    Additional postprocessing should be performed on the returned values.

    Parameters:
    ----------
    contest_id : str
        The ID of the Codeforces contest.
    problem_index : str
        The index of the problem within the contest (e.g., 'A', 'B', 'C', etc.).
    verdict_name : str
        The name of the verdict to filter submissions (e.g., CodeforcesVerdict.ACCEPTED, etc.).
    language : str
        The programming language for the filter (e.g., CodeforcesLanguage.PYTHON3, etc.).

    Returns:
    -------
    list of str:
        A list of submission info dicts that match the filter.

    """
    submissions = []
    page = 1
    with set_cf_filter(contest_id, problem_index, verdict_name, language) as session:
        while True:
            # Fetch the current submissions page
            page_submissions, max_pages = crawl_submissions_page(session, contest_id, problem_index, page)

            submissions.extend(page_submissions)
            page += 1

            should_stop = stop_criteria(submissions) if stop_criteria is not None else False
            if page > max_pages or should_stop:
                # If we have satisfied the custom stopping criteria or reached the last page, stop fetching.
                break

    return submissions


@sleep_after_execution(10)
def crawl_submissions(contest_id, problem_index, language="python"):
    def build_submission_dict(codes, metas):
        assert len(codes) == len(metas)
        return [dict(meta=meta, code=code) for meta, code in zip(metas, codes)]

    # Override settings
    if language == "python":
        language = CodeforcesLanguage.PYPY_3
        # Used for crawling accepted submissions - there are not many, so check in different "languages"
        language_group = [CodeforcesLanguage.PYPY_3,
                          CodeforcesLanguage.PYPY_3_64,
                          CodeforcesLanguage.PYTHON_3
                          ]
    elif language == "cpp":
        language = CodeforcesLanguage.GNU_CPP_14
        # Used for crawling accepted submissions - there are not many, so check in different "languages"
        language_group = [CodeforcesLanguage.GNU_CPP_14,
                          CodeforcesLanguage.GNU_CPP_17,
                          CodeforcesLanguage.GNU_CPP_17_64,
                          CodeforcesLanguage.GNU_CPP_20_64,
                          CodeforcesLanguage.CLANG_PP_17_DIAGNOSTICS,
                          CodeforcesLanguage.CLANG_PP_20_DIAGNOSTICS,
                          CodeforcesLanguage.MS_CPP_2017,
                          ]
    else:
        raise NotImplementedError(f"Language {language} is not supported.")

    # Values to return
    submissions = dict()

    # Crawl ACCEPTED

    acc_max_count = 5
    criteria_acc_max_count = lambda li: len(li) >= acc_max_count
    accepted_submissions = []
    for lan in language_group:
        tmp_acc_submissions = find_submission_ids(contest_id, problem_index,
                                                  verdict_name=CodeforcesVerdict.ACCEPTED,
                                                  language=lan,
                                                  stop_criteria=criteria_acc_max_count)
        accepted_submissions.extend(tmp_acc_submissions)
        if criteria_acc_max_count(accepted_submissions):
            break
    if len(accepted_submissions) == 0:
        logger.warning(f"No accepted submissons found for problem {contest_id}_{problem_index}.")
    accepted_submissions = accepted_submissions[:acc_max_count]
    acc_submission_codes = [crawl_submission_code(contest_id, submission["submission_id"]) for submission in
                            accepted_submissions]
    acc_submission_dict = build_submission_dict(metas=accepted_submissions, codes=acc_submission_codes)
    submissions[CodeforcesVerdict.ACCEPTED] = acc_submission_dict

    # Crawl WRONG_ANSWER
    wa_max_count = 300
    criteria_wa_max_count = lambda li: len(li) >= wa_max_count or len(set([item["verdict"] for item in li])) >= 10
    wa_submissions = []
    for lan in language_group:
        tmp_wa_submissions = find_submission_ids(contest_id, problem_index,
                                                 verdict_name=CodeforcesVerdict.WRONG_ANSWER,
                                                 language=lan,
                                                 stop_criteria=criteria_wa_max_count)
        wa_submissions.extend(tmp_wa_submissions)
        if criteria_wa_max_count(wa_submissions):
            break
    # Only keep a maximum of 10 diverse WA answers, by selecting submissions failing on different tests
    wa_submissions = pick_diverse_elements(wa_submissions, key="verdict", max_count=10)
    wa_submission_codes = [crawl_submission_code(contest_id, submission["submission_id"]) for submission in
                           wa_submissions]
    wa_submission_dict = build_submission_dict(metas=wa_submissions, codes=wa_submission_codes)
    submissions[CodeforcesVerdict.WRONG_ANSWER] = wa_submission_dict

    # Crawl rejected miscs: RUNTIME_ERROR, TIME_LIMIT_EXCEEDED, MEMORY_LIMIT_EXCEEDED,
    misc_verdicts = [
        CodeforcesVerdict.RUNTIME_ERROR,
        CodeforcesVerdict.TIME_LIMIT_EXCEEDED,
        CodeforcesVerdict.MEMORY_LIMIT_EXCEEDED
    ]
    misc_max_count = 3
    criteria_misc_max_count = lambda li: len(li) >= misc_max_count
    for verdict in misc_verdicts:
        misc_submissions = find_submission_ids(contest_id, problem_index,
                                               verdict_name=verdict,
                                               language=language,
                                               stop_criteria=criteria_misc_max_count)
        misc_submissions = misc_submissions[:misc_max_count]
        misc_submission_codes = [crawl_submission_code(contest_id, submission["submission_id"]) for submission in
                                 misc_submissions]
        misc_submission_dict = build_submission_dict(metas=misc_submissions, codes=misc_submission_codes)
        submissions[verdict] = misc_submission_dict

    # Crawl CHALLENGED
    hacked_max_count = 3
    criteria_hacked_max_count = lambda li: len(li) >= hacked_max_count
    hacked_submissions = find_submission_ids(contest_id, problem_index,
                                             verdict_name=CodeforcesVerdict.CHALLENGED,
                                             language=language,
                                             stop_criteria=criteria_hacked_max_count)
    hacked_submissions = hacked_submissions[:hacked_max_count]
    hacked_submission_codes = [crawl_submission_code(contest_id, submission["submission_id"]) for submission in
                               hacked_submissions]
    hacked_submission_dict = build_submission_dict(metas=hacked_submissions, codes=hacked_submission_codes)
    submissions[CodeforcesVerdict.CHALLENGED] = hacked_submission_dict

    return submissions


def find_saved_list(crawled_root_folder):
    crawled_list = []

    for folder_name in os.listdir(crawled_root_folder):  # folder_name is basename
        folder_path = os.path.join(crawled_root_folder, folder_name)
        if os.path.isdir(folder_path) and folder_name.startswith("cf_"):
            # Extract contest_id and problem_index from the folder name e.g. "cf_123_A"
            contest_id, problem_index = folder_name.split("_")[1:3]
            crawled_list.append({"contest_id": contest_id, "problem_index": problem_index})

    return crawled_list


def get_problems_with_images(cf_problems_folder):
    """
    Scan the image folder to retrieve problems with images.
    """
    # Create an empty list to store the tuples
    problems = []

    # Get all subdirectories with image folder
    subdirs = [d for d in os.listdir(cf_problems_folder)
               if os.path.isdir(os.path.join(cf_problems_folder, d)) and os.path.exists(
            os.path.join(cf_problems_folder, d, "images"))]

    for dir in subdirs:
        contest_id, problem_index = dir.split('_')
        problems.append({
            "contest_id": contest_id,
            "problem_index": problem_index
        })

    # Return the list of tuples
    return problems


def save_problems():
    # Fetch and save the problemset
    problems = fetch_problemset()

    for problem in tqdm.tqdm(problems):
        try:
            scrape_problem(problem['contest_id'], problem['problem_index'])
        except:
            logger.error(f"Error scrawling problem {problem['contest_id'], problem['problem_index']}")
        time.sleep(5)


def clean_saved_submissions_and_tests(save_dir):
    problems = find_saved_list(save_dir)
    for problem in problems:
        contest_id, problem_index = problem['contest_id'], problem['problem_index']
        test_cases_file = os.path.join(save_dir, f"cf_{contest_id}_{problem_index}", "input_output.json")
        if os.path.exists(test_cases_file):
            os.remove(test_cases_file)

    submissions_folder = os.path.join(save_dir, "submissions")
    if os.path.exists(submissions_folder):
        for verdict_file in os.listdir(submissions_folder):
            file_path = os.path.join(submissions_folder, verdict_file)
            if os.path.exists(file_path):
                os.remove(file_path)

        os.rmdir(submissions_folder)
    logger.info("Cleaned saved submissions and test cases.")


def save_submissions_and_tests(save_dir, problems, progress_file=None, language="python"):
    def write_progress(problem, progress_file):
        contest_id, problem_index = problem['contest_id'], problem['problem_index']
        with open(progress_file, 'a') as f:
                    f.write(f"{contest_id}_{problem_index}\n")
    # Try to resume crawling
    if progress_file and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            saved_problems = set([item.strip() for item in f.readlines()])
        logger.info(f"{len(saved_problems)} problems already saved. Skipping them.")
        # Filter crawled
        problems = list(filter(lambda x: f"{x['contest_id']}_{x['problem_index']}" not in saved_problems, problems))
        # Filter skip list
        problems = list(filter(lambda x: f"{x['contest_id']}_{x['problem_index']}" not in SKIP_LIST, problems))

    pbar_problems = tqdm.tqdm(problems)
    for problem in pbar_problems:
        contest_id, problem_index = problem['contest_id'], problem['problem_index']
        pbar_problems.set_description(f"Crawling {contest_id}_{problem_index}")
        # Crawl data
        test_cases = crawl_tests(contest_id, problem_index)
        # Error occurred. Skip the problem.
        if test_cases is None:
            write_progress(problem, progress_file)
            logger.warning(f"Skipping {problem['contest_id']}_{problem['problem_index']} as attempts to get test cases failed."
                            "It is likely that there is no access to the submissions, e.g. in ICPC contests.")
            continue

        submissions = crawl_submissions(contest_id, problem_index, language=language)

        # Save data
        if len(test_cases) > 0:
            test_cases_file = os.path.join(save_dir, f"cf_{contest_id}_{problem_index}", "input_output.json")
            with open(test_cases_file, 'w') as f:
                f.write(json.dumps(test_cases, indent=2))

        submissions_folder = os.path.join(save_dir, f"cf_{contest_id}_{problem_index}", "submissions")
        os.makedirs(submissions_folder, exist_ok=True)
        existing_verdicts = list(filter(lambda key: len(submissions[key]) > 0, submissions.keys()))
        for verdict in existing_verdicts:
            output_path = os.path.join(submissions_folder, f"{verdict}.json")
            with open(output_path, 'w') as f:
                f.write(json.dumps(submissions[verdict], indent=2))

        logger.info(f"Successfully crawled {contest_id}_{problem_index}")
        write_progress(problem, progress_file)


def save_submissions_by_lang(save_dir, problems, language="cpp", progress_file=None):
    def write_progress(problem, progress_file):
        contest_id, problem_index = problem['contest_id'], problem['problem_index']
        with open(progress_file, 'a') as f:
                    f.write(f"{contest_id}_{problem_index}\n")
    # Try to resume crawling
    if progress_file and os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            saved_problems = set([item.strip() for item in f.readlines()])
        logger.info(f"{len(saved_problems)} problems already saved. Skipping them.")
        # Filter crawled
        problems = list(filter(lambda x: f"{x['contest_id']}_{x['problem_index']}" not in saved_problems, problems))
        # Filter skip list
        problems = list(filter(lambda x: f"{x['contest_id']}_{x['problem_index']}" not in SKIP_LIST, problems))

    pbar_problems = tqdm.tqdm(problems)
    for problem in pbar_problems:
        contest_id, problem_index = problem['contest_id'], problem['problem_index']
        pbar_problems.set_description(f"Crawling {contest_id}_{problem_index}")

        submissions = crawl_submissions(contest_id, problem_index, language=language)

        submissions_folder = os.path.join(save_dir, f"cf_{contest_id}_{problem_index}", "submissions", language)
        os.makedirs(submissions_folder, exist_ok=True)
        existing_verdicts = list(filter(lambda key: len(submissions[key]) > 0, submissions.keys()))
        for verdict in existing_verdicts:
            output_path = os.path.join(submissions_folder, f"{verdict}.json")
            with open(output_path, 'w') as f:
                f.write(json.dumps(submissions[verdict], indent=2))

        logger.info(f"Successfully crawled {contest_id}_{problem_index}")
        write_progress(problem, progress_file)


def main():
    save_problems()

    # save_dir = os.path.join("crawled", "codeforces", "problems")
    # problems = sorted(find_saved_list(save_dir), key=lambda x: (int(x['contest_id']), x['problem_index']))
    # problems = sorted(find_saved_list("/home/kaixin/new_temp"), key=lambda x: (int(x['contest_id']), x['problem_index']))
    # problems = list(filter(lambda x: int(x['contest_id']) >= 1850, problems))
    # # clean_saved_submissions_and_tests(save_dir)
    # save_submissions_and_tests(save_dir, problems, progress_file=os.path.join("crawled", "codeforces", "progress.txt"))
    # save_submissions_by_lang(save_dir, problems, progress_file=os.path.join("crawled", "codeforces", "progress_new.txt"), language="python")


if __name__ == "__main__":
    main()
