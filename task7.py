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
import heapq

# Pearson correlation and p-values

# Go through each utterance, and for each dialogue act, count the number of occurances of each
# emotion, and sum of compound sentiment
# Then for each dialogue act, whichever emotion appeared the most, the best it correlates with
# that dialogue act. Same for sentiment. Calculate the overall sentiment for all utterances with
# that dialogue act

def main():

    # Make a command line ui to input filenames for data
    dialogue_acts_data = 'all_dialogues_predictions.json'
    emotions = 'emos.json'
    sentiments = 'sentiments.json'

    with open(dialogue_acts_data, 'r', encoding='utf-8') as file:
        dialogue_acts_data = json.load(file)

    with open(emotions, 'r', encoding='utf-8') as file:
        emotions = json.load(file)

    with open(sentiments, 'r', encoding='utf-8') as file:
        sentiments = json.load(file)

    # The first element in the list holds a dictionary that has emotions as keys and the number of times that
    # emotion has appeared in the same utterance for that dialogue act as values
    # The second element in the list holds the compound sentiment and third keeps track of how many compound
    # sentiments were added together
    correlations = {
        "Emotion": [{}, 0, 0],
        "yAnswer": [{}, 0, 0],
        #"yAnswer" : [{}, 0, 0],
        "Continuer": [{}, 0, 0],
        "whQuestion": [{}, 0, 0],
        "System": [{}, 0, 0],
        "Accept": [{}, 0, 0],
        "Clarify": [{}, 0, 0],
        #"Clarity": [{}, 0, 0],
        "Emphasis": [{}, 0, 0],
        "nAnswer": [{}, 0, 0],
        "Greet": [{}, 0, 0],
        "Statement": [{}, 0, 0],
        "Reject": [{}, 0, 0],
        "Bye": [{}, 0, 0],
        "Other" : [{}, 0, 0],
        #"Others" : [{}, 0, 0],
        "ynQuestion" : [{}, 0, 0],
    }

    # Iterate over each utterance in each dialogue
    for dialogue in range(len(dialogue_acts_data)):
        for utterance in range(len(dialogue_acts_data[dialogue])):

            
            dialogue_act = dialogue_acts_data[dialogue][utterance]
            sentiment = sentiments[dialogue][utterance]

            # If emotions list is empty, then emotions[dialogue][utterance][0] doens't exist,
            # but if it has an emotion, then emotions[dialogue][utterance] is a list and not a string
            if len(emotions[dialogue][utterance]) != 0:
                emotion = emotions[dialogue][utterance][0]
            else:
                emotion = 'NaN'

            # Adds emotion or increments emotion value in the correlations dict lists first dictionary
            # If emotion is not found before for this dialogue act, it is added to the dict as a key, with a value of one.
            # Otherwise the value for that key is incremented by one
            # If emotion is nan, do nothing
            if emotion == 'NaN':
                a = 1
            elif emotion in correlations[dialogue_act][0]:
                correlations[dialogue_act][0][emotion] += 1
            else:
                correlations[dialogue_act][0][emotion] = 1

            # Adds compound sentiment to the correlations dict lists second element
            correlations[dialogue_act][1] += sentiment["compound"]
            correlations[dialogue_act][2] += 1

    # This list stores lists, which have two elements. First is a dialogue act, second is the emotion that dialogue act has
    # the highest correlation with, i.e. they appear the most together
    highest_emotion_correlations = []

    for dialogue_act in correlations:
        
        # Get average compound sentiment
        correlations[dialogue_act][1] = correlations[dialogue_act][1] / correlations[dialogue_act][2]
        print("Compound sentiment for " + str(dialogue_act) + " is: " + str(correlations[dialogue_act][1]))

        # For each dialogue act, find emotion that appears the most with it
        if len(correlations[dialogue_act][0]) != 0:
            highest_emotion_correlations.append([dialogue_act, max(correlations[dialogue_act][0], key=correlations[dialogue_act][0].get)])

    print(highest_emotion_correlations)

if __name__ == "__main__":
    main()