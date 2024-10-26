# NLP Project 16
# 26.10.2024
# Eetu Hyypiö

import pandas as pd
import numpy as np
import openpyxl
import xlsxwriter

# Extracts all line numbers of lines in the specified topic. The topic_number argument must be given as a string. For example: '9'
# Topic number 9 is politics
def save_topic_lines(path_to_topic_file, topic_number):

    topic_lines = []

    with open(path_to_topic_file, 'r') as file:
        
        i = 1

        for line in file:
            if line[0] == topic_number:
                topic_lines.append(i)
            
            i = i + 1

    return topic_lines

# Extracts all dialogue lines from a specific topic
# if topic is 'all', every topic is extracted
def extract_topic(path_to_dialogue_file, path_to_topic_file, topic_number):

    topic_lines = save_topic_lines(path_to_topic_file, topic_number)
    topic_dialogue = []

    with open(path_to_dialogue_file, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file):

            if topic_number == 'all':
                topic_dialogue.append(line)

            elif topic_number != 'all':
                if line_number in topic_lines:
                    topic_dialogue.append(line)

    return topic_dialogue

# Creates a pandas dataframe for the dialogue data in a specific topic
# Rows are dialogue lines. They are in the same order as in the original dialogues_text.txt file
# Columns are utterances in that dialogue.
def create_topic_dataframe(path_to_dialogue_file, path_to_topic_file, topic_number):

    topic_dialogue = extract_topic(path_to_dialogue_file, path_to_topic_file, topic_number)
    split_dialogue = [line.split('__eou__') for line in topic_dialogue]
    topic_dialogue_data = pd.DataFrame(split_dialogue)

    return topic_dialogue_data

# Saves the dataframe in excel format
# This is just for not having to write the annoying file format
def save_dataframe_as_excel(data, filename):

    if '.xlsx' not in filename:
        filename = filename + '.xlsx'

    data.to_excel(filename, header=False, index=False)

# Does everything above. Extracts the topic, makes it into a dataframe and saves in excel format
# if topic number is 'all', every topic is extracted
def extract_and_save_topic_dialogue(path_to_dialogue_file, path_to_topic_file, topic_number, filename):

    topic_dialogue_data = create_topic_dataframe(path_to_dialogue_file, path_to_topic_file, topic_number)
    save_dataframe_as_excel(topic_dialogue_data, filename)


def main():

    print("Give topic number to be saved as an excel file, and the name of the file. 'all' is for saving all topics")
    topic_number = input("Topic: ")
    name_of_file = input("Name of file: ")

    extract_and_save_topic_dialogue('ijcnlp_dailydialog/dialogues_text.txt', 'ijcnlp_dailydialog/dialogues_topic.txt', topic_number, name_of_file)



if __name__ == "__main__":
    main()