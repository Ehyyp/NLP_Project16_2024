# NLP Project 16
# 28.10.2024
# Eetu Hyypiö

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import spacy
from task1_stats import count_utterances
from task1_stats import open_process_data

nltk.download('averaged_perceptron_tagger_eng')

# Opens an excel data file saved in the format that task1_save_topic.py saves and
# calculates the entity tags per utterance
def avg_person_organization_entity_tags_per_utterance(excel_file_name):

    entity_tagger = spacy.load("en_core_web_md")
    dialogues = open_process_data(excel_file_name)
    num_entities = 0

    for dialogue in dialogues:
        for utterance in dialogue:
            entity_tagged_utterance = entity_tagger(utterance)

            for entity in entity_tagged_utterance.ents:
                if entity.label_ == ("ORG" or "PERSON"):
                    num_entities += 1

    num_utterances = count_utterances(dialogues)
    avg_ent_tag_per_utterance = num_entities / num_utterances

    return avg_ent_tag_per_utterance

def main():

    print("Input the name of the excel data file from which to calculate the average number of person/organization named-entities per utterance")
    excel_file_name = input("File name: ")

    if '.xlsx' not in excel_file_name:
        excel_file_name = excel_file_name + '.xlsx'

    ent_tags_avg = avg_person_organization_entity_tags_per_utterance(excel_file_name)
    print("Average number of person/organization named-entities per utterance: " + str(ent_tags_avg))

if __name__ == "__main__":
    main()