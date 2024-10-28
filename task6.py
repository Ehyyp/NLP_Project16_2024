# NLP Project 16
# Eetu Hyypiö
# 28.10.2024

import nltk
from task1_stats import tokenize_data
from task1_save_topic import create_topic_dataframe
import pandas as pd
import openpyxl
import json
import string

# Opens excel data and saves it into a multidimensional list
# Use task1_save_topic.py to save topic dialogues in this excel form
def open_process_data(name_of_data_file):
    
    data = pd.read_excel(name_of_data_file)
    rows, columns = data.shape
    dialogs = []
    
    for row in range(rows):
        utterances = []

        for column in range(columns):
            if type(data.iat[row, column]) is str:
                utterances.append(data.iat[row, column])
            
        dialogs.append(utterances)

    return dialogs

# Feature extraction function as specified in the NLTK organization book chapter 6, section 2.2
def dialogue_act_features(post):
    features = {}
    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True
    return features

# Trains the classifier as specified in the NLTK organization book chapter 6, section 2.2
def train_NLTK_model():

    nltk.download('nps_chat')
    posts = nltk.corpus.nps_chat.xml_posts()[:10000]

    featuresets = [(dialogue_act_features(post.text), post.get('class')) for post in posts]
    size = int(len(featuresets) * 0.1)
    train_set, test_set = featuresets[size:], featuresets[:size]
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    print("Classifier accuracy on the NLTK NPS corpus: " + str(nltk.classify.accuracy(classifier, test_set)))

    return classifier

# Classifies data from an excel file
def classify_data(name_of_data_file):

    data = open_process_data(name_of_data_file)
    classifier = train_NLTK_model()

    predictions = []

    for dialog in data:
        dialog_predictions = []

        for utterance in dialog:
            utterance_features = dialogue_act_features(utterance)
            prediction = classifier.classify(utterance_features)
            dialog_predictions.append(prediction)

        predictions.append(dialog_predictions)

    return predictions
    
# Saves predictions
def save_to_json(predictions, saved_file_name):

    with open(saved_file_name, "w") as file:
        json.dump(predictions, file, indent=4)


def main():
    print("Input name of excel data file to be classified")
    name_of_excel_file = input("Excel file name: ")
    print("Input name of json file where to save predictions")
    name_of_json_file = input("Json file name: ")

    if '.xlsx' not in name_of_excel_file:
        name_of_excel_file = name_of_excel_file + '.xlsx'

    if '.json' not in name_of_json_file:
        name_of_json_file = name_of_json_file + '.json'

    predictions = classify_data(name_of_excel_file)
    save_to_json(predictions, name_of_json_file)


if __name__ == "__main__":
    main()
