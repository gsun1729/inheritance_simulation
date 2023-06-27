from typing import Union
import os


def printProgressBar (iteration, total, prefix = 'Progress:', suffix = 'Complete', decimals = 1, length = 50, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
        
    Source: 
        https://stackoverflow.com/questions/3173320/text-progress-bar-in-terminal-with-block-characters
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
        
def getFilenames(root_directory : str, queries : Union[list,str] = '.tif'):
    """Given directory with datafiles, get just imagefilenames

    :param root_directory: root search directory.  Function will traverse subdirectories for image files, 
        so ideally if you only want images within the same level of directory, don't use this function
    :type root_directory: str

    :param queries: strings to look for in the filename
    :type queries: str, List[str]

    :rtype: List[str]
        a list of file paths to files that match that of the regex query
    """
    img_filelist = []
    for current_location, sub_directories, files in os.walk(root_directory):
        if files:
            for img_file in files:
                if isinstance(queries, str):
                    if (queries.lower() in img_file.lower()) and '_thumb_' not in img_file:
                        img_filelist.append(os.path.join(current_location, img_file))
                elif isinstance(queries, list):
                    if all(query.lower() in img_file.lower() for query in queries) and '_thumb_' not in img_file:
                        img_filelist.append(os.path.join(current_location, img_file))
    img_filelist.sort()
    return img_filelist