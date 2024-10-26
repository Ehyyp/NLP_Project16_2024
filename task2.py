# NLP Project 16
# 26.10.2024
# Eetu Hyypiö

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
import spacy
from task1_save_topic import create_topic_dataframe
from task1_stats import count_utterances

nltk.download('averaged_perceptron_tagger_eng')

def avg_person_organization_entity_tags_per_utterance(dataframe):

    entity_tagger = spacy.load("en_core_web_md")

    rows, columns = dataframe.shape
    num_entities = 0

    for row in range(rows):
        for column in range(columns):
            if type(dataframe.iat[row, column]) is str:
                entity_tagged_text = entity_tagger(dataframe.iat[row, column])

                for entity in entity_tagged_text.ents:
                    if entity.label_ == ("ORG" or "PERSON"):
                        num_entities = num_entities + 1

    num_utterances = count_utterances(dataframe)
    avg_ent_tag_per_utterance = num_entities / num_utterances

    return avg_ent_tag_per_utterance


def main():

    print("Give the topic number for which to calculate the average number of person/organization named-entities per utterance. 'all' is for all dialogue")
    topic_number = input("Topic: ")

    dialogue_data = create_topic_dataframe('ijcnlp_dailydialog/dialogues_text.txt', 'ijcnlp_dailydialog/dialogues_topic.txt', topic_number)
    ent_tags_avg = avg_person_organization_entity_tags_per_utterance(dialogue_data)
    print("Average number of person/organization named-entities per utterance: " + str(ent_tags_avg))



if __name__ == "__main__":
    main()