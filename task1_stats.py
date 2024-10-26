# NLP Project 16
# 26.10.2024
# Eetu Hyypiö

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string
from task1_save_topic import create_topic_dataframe

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')

# Takes into dataframe and concatenates everything in it to be a single string. This is for tokenization and such
def form_dialogue_string(dataframe):

    dialogue_string = ''
    rows, columns = dataframe.shape

    for row in range(rows):
        for column in range(columns):
            if type(dataframe.iat[row, column]) is str:
                dialogue_string = dialogue_string + dataframe.iat[row, column]

    return dialogue_string


# Remove punctuation, lowercase and tokenize
# There still remains things like "sure.it" and "t", remove
def preprocess_dialogue(dialogue):

    stop = set(list(string.punctuation))

    tokenized = word_tokenize(dialogue.lower())
    processed_dialogue = [word for word in tokenized if word not in stop]

    return processed_dialogue


# Calculates the vocabulary size for a dataframe
def vocabulary_size(dataframe):
    
    dialogue_string = form_dialogue_string(dataframe)
    processed_dialogue = preprocess_dialogue(dialogue_string)
    unique_tokens = set(processed_dialogue)
    vocabulary_size = len(unique_tokens)

    return vocabulary_size


# Calculates the number of utterances for a dataframe
def count_utterances(dataframe):

    num_of_utterances = 0
    rows, columns = dataframe.shape

    for row in range(rows):
        for column in range(columns):
            if type(dataframe.iat[row, column]) is str:
                num_of_utterances = num_of_utterances + 1

    return num_of_utterances


# Count average tokens per utterance from a dataframe
def count_avg_tokens_per_utterance(dataframe):

    num_of_utterances = count_utterances(dataframe)

    dialogue_string = form_dialogue_string(dataframe)
    processed_dialogue = preprocess_dialogue(dialogue_string)

    avg_tokens_per_utterance = len(processed_dialogue) / num_of_utterances

    return avg_tokens_per_utterance


# Uses NLTK part of speech tagger to identify pronouns, counts
# the number of pronouns and then the average per utterance
def avg_pronouns_per_utterance(dataframe):

    dialogue_string = form_dialogue_string(dataframe)
    processed_dialogue = preprocess_dialogue(dialogue_string)

    tagged_dialogue = pos_tag(processed_dialogue)
    
    pronoun_count = 0

    for (token, prp_tag) in tagged_dialogue:
        if prp_tag == ('PRP' or 'PRP$'):
            pronoun_count = pronoun_count + 1

    num_of_utterances = count_utterances(dataframe)
    avg_prp = pronoun_count / num_of_utterances

    return avg_prp


# Didn't find any clear resource for agreement or negation wording.
# There is nltk.metrics.agreement, but it is not for counting agreement words
# There is also the option to try and find negation/agreement related words through wordnet, but it would also find words that are not specifially negation/agreement words
# The custom list of agreement/negation words is subject to change
# choice = 1 counts average number of agreement words
# choice = 2 does the same for negation words
def avg_agreement_negation_per_utterance(dataframe, choice):

    agreement_words = ['yes', 'ok', 'sure', 'okay', 'agreed', 'agree']
    negation_words = ['no', 'not', "don't", "can't", 'neither', ]

    if choice == 1:
        words_to_count = agreement_words
    elif choice == 2:
        words_to_count = negation_words
    else:
        print("Second argument: 1 for agreement words, 2 for negation words")
        return 0

    dialogue_string = form_dialogue_string(dataframe)
    processed_dialogue = preprocess_dialogue(dialogue_string)
    num_of_utterances = count_utterances(dataframe)

    num_words_to_count = 0

    for word in processed_dialogue:
        if word in words_to_count:
            num_words_to_count = num_words_to_count + 1
    
    avg_agreement = num_words_to_count / num_of_utterances

    return avg_agreement

# Prints all stats for a given topic
def create_stats_table(path_to_dialogue_file, path_to_topic_file, topic_number):

    topic_data = create_topic_dataframe(path_to_dialogue_file, path_to_topic_file, topic_number)

    vocab = vocabulary_size(topic_data)
    utterances = count_utterances(topic_data)
    tokens_per_utterance = count_avg_tokens_per_utterance(topic_data)
    avg_prp = avg_pronouns_per_utterance(topic_data)
    avg_agreement = avg_agreement_negation_per_utterance(topic_data, 1)
    avg_negation = avg_agreement_negation_per_utterance(topic_data, 2)

    print("Stats for topic number " + topic_number + ":")
    print("Size of vocabulary: " + str(vocab))
    print("Number of utterances: " + str(utterances))
    print("Average number of tokens per utterance: " + str(tokens_per_utterance))
    print("Average number of pronouns per utterance: " + str(avg_prp))
    print("Average number of agreement words per utterance: " + str(avg_agreement))
    print("Average number of negation words per utterance: " + str(avg_negation))


def main():

    print("Give the topic number for which to calculate stats. 'all' calculates stats for all dialogue")
    topic_number = input("Topic: ")

    create_stats_table('ijcnlp_dailydialog/dialogues_text.txt', 'ijcnlp_dailydialog/dialogues_topic.txt', topic_number)



if __name__ == "__main__":
    main()