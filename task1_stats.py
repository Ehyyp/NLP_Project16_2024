# NLP Project 16
# 28.10.2024
# Eetu Hyypiö

import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import string

nltk.download('stopwords')
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')

# Open the data that was made with task1_save_topic.py
def open_process_data(name_of_excel_file):
    
    data = pd.read_excel(name_of_excel_file)
    rows, columns = data.shape
    dialogues = []
    
    for row in range(rows):

        utterances = []

        for column in range(columns):
            if type(data.iat[row, column]) is str:
                utterances.append(data.iat[row, column])
            
        dialogues.append(utterances)

    return dialogues

# Tokenizes, lowers and removes special characters from data
def tokenize_data(dialogues):
    
    special = ['!', '@', '#', '$', '%', '^', '&', '*', '(', ')', '-', '=', '+', '[', ']', '{', '}', ';', ':', '"', "'", '<', '>', ',', '.', '/', '?', '\\', '|', '`', '~', '...']
    tokenized_dialogues = []

    for dialogue in dialogues:
        tokenized_dialogue = []

        for utterance in dialogue:
            processed_tokenized_utterance = []
            tokenized_utterance = word_tokenize(utterance)

            for token in tokenized_utterance:
                token.lower()
                if (token not in special) and (len(token) != 1):
                    processed_tokenized_utterance.append(token)
                
            tokenized_dialogue.append(processed_tokenized_utterance)
        
        tokenized_dialogues.append(tokenized_dialogue)
    
    return tokenized_dialogues

# Calculates the vocabulary size for data. Data is given in the form that tokenize_data returns it
def vocabulary_size(dialogues):
    
    tokenized_dialogues = tokenize_data(dialogues)
    counted_words = []
    vocabulary_size = 0

    for dialogue in tokenized_dialogues:
        for utterance in dialogue:
            for token in utterance:
                if token not in counted_words:
                    vocabulary_size += 1
                    counted_words.append(token)
                    #print("Unique token: " + token)

    return vocabulary_size

# Calculates the number of utterances for a dialogue
def count_utterances(dialogues):

    num_of_utterances = 0

    for dialogue in dialogues:
        for utterance in dialogue:
            num_of_utterances += 1

    return num_of_utterances

# Count average tokens per utterance for dialogue
def count_avg_tokens_per_utterance(dialogues):

    num_of_utterances = count_utterances(dialogues)
    tokenized_dialogues = tokenize_data(dialogues)
    total_tokens = 0

    for dialogue in tokenized_dialogues:
        for utterance in dialogue:
            total_tokens += len(utterance)

    avg_tokens_per_utterance = total_tokens / num_of_utterances

    return avg_tokens_per_utterance


# Uses NLTK part of speech tagger to identify pronouns, counts
# the number of pronouns and then the average per utterance
def avg_pronouns_per_utterance(dialogues):
    
    tokenized_dialogues = tokenize_data(dialogues)
    pronoun_count = 0

    for dialogue in tokenized_dialogues:
        for utterance in dialogue:
            tagged_utterance = pos_tag(utterance)

            for (token, prp_tag) in tagged_utterance:
                if prp_tag == ('PRP' or 'PRP$'):
                    pronoun_count += 1

    num_of_utterances = count_utterances(dialogues)
    avg_prp = pronoun_count / num_of_utterances

    return avg_prp


# Didn't find any clear resource for agreement or negation wording.
# There is nltk.metrics.agreement, but it is not for counting agreement words
# There is also the option to try and find negation/agreement related words through wordnet, but it would also find words that are not specifially negation/agreement words
# The custom list of agreement/negation words is subject to change
# choice = 1 counts average number of agreement words
# choice = 2 does the same for negation words
def avg_agreement_negation_per_utterance(dialogues, choice):

    agreement_words = ['yes', 'ok', 'sure', 'okay', 'agreed', 'agree']
    negation_words = ['no', 'not', "don't", "can't", 'neither', ]

    if choice == 1:
        words_to_count = agreement_words
    elif choice == 2:
        words_to_count = negation_words
    else:
        print("Second argument: 1 for agreement words, 2 for negation words")
        return 0

    tokenized_dialogues = tokenize_data(dialogues)
    num_of_utterances = count_utterances(dialogues)
    num_words_to_count = 0

    for dialogue in tokenized_dialogues:
        for utterance in dialogue:
            for token in utterance:
                if token in words_to_count:
                    num_words_to_count = num_words_to_count + 1
    
    avg_agreement_negation = num_words_to_count / num_of_utterances

    return avg_agreement_negation

# Prints all stats for a given topic
def print_stats_from_excel(name_of_excel_file):

    dialogues = open_process_data(name_of_excel_file)
    vocab = vocabulary_size(dialogues)
    utterances = count_utterances(dialogues)
    tokens_per_utterance = count_avg_tokens_per_utterance(dialogues)
    avg_prp = avg_pronouns_per_utterance(dialogues)
    avg_agreement = avg_agreement_negation_per_utterance(dialogues, 1)
    avg_negation = avg_agreement_negation_per_utterance(dialogues, 2)

    print("Stats for file \"" + name_of_excel_file + "\":")
    print("Size of vocabulary: " + str(vocab))
    print("Number of utterances: " + str(utterances))
    print("Average number of tokens per utterance: " + str(tokens_per_utterance))
    print("Average number of pronouns per utterance: " + str(avg_prp))
    print("Average number of agreement words per utterance: " + str(avg_agreement))
    print("Average number of negation words per utterance: " + str(avg_negation))

def main():

    print("Give name of data excel file for which to calculate stats")
    name_of_excel_file = input("File name: ")

    if '.xlsx' not in name_of_excel_file:
        name_of_excel_file = name_of_excel_file + '.xlsx'

    print_stats_from_excel(name_of_excel_file)

if __name__ == "__main__":
    main()