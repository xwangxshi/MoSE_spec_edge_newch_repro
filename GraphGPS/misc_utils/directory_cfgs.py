import os

def list_files_in_directory(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            print(os.path.abspath(os.path.join(root, file)))


directory = ""
list_files_in_directory(directory)
