
import os

def write_tuples_to_file(array_of_tuples, subdirectory, file_name):
    # See if there is a better way to do this
    cwd = os.getcwd()
    file_path = os.path.join(subdirectory, file_name)
    with open(cwd + file_path, 'w') as file:
        for tpl in array_of_tuples:
            line = ' '.join(map(str, tpl)) + '\n'
            file.write(line)