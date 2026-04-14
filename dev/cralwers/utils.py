import os
import time


def find_crawled_list(crawled_root_folder):
    crawled_list = []

    for folder_name in os.listdir(crawled_root_folder):  # folder_name is basename
        folder_path = os.path.join(crawled_root_folder, folder_name)
        if os.path.isdir(folder_path):
            crawled_list.append(folder_name)

    return crawled_list


def get_problems_with_images(problems_folder):
    """
    Scan the image folder to retrieve problems with images.
    """
    # Create an empty list to store the tuples
    problems = []

    # Get all subdirectories with image folder
    subdirs = [d for d in os.listdir(problems_folder) 
               if os.path.isdir(os.path.join(problems_folder, d)) and os.path.exists(os.path.join(problems_folder, d, "images"))]

    # Return the list of strs
    return subdirs

def sleep_after_execution(seconds):
    def decorator(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            time.sleep(seconds)
            return result
        return wrapper
    return decorator