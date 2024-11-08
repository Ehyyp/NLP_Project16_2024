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

# Calculates the emotion most associated with each dialogue act and the average compound sentiment of each dialogue act
def calculate_matching(dialogue_acts_data_file, emotions_file, sentiments_file):

    with open(dialogue_acts_data_file, 'r', encoding='utf-8') as file:
        dialogue_acts_data = json.load(file)

    with open(emotions_file, 'r', encoding='utf-8') as file:
        emotions = json.load(file)

    with open(sentiments_file, 'r', encoding='utf-8') as file:
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

    return correlations, highest_emotion_correlations

# The data in dialogue act dataframe is converted to be numerical in this format
# none = 0
#"Emotion" = 1
#"yAnswer" = 2
#"Continuer" = 3
#"whQuestion" = 4
#"System" = 5
#"Accept" = 6
#"Clarify" = 7
#"Emphasis" = 8
#"nAnswer" = 9
#"Greet" = 10
#"Statement" = 11
#"Reject" = 12
#"Bye" = 13
#"Other" = 14
#"ynQuestion" = 15
def calculate_correlations(dialogue_acts_data_file, emotions_file, sentiments_file):

    # Load data into dataframes
    dialogue_acts_dataframe = pd.read_json(dialogue_acts_data_file)
    emotions_dataframe = pd.read_json(emotions_file)
    sentiments_dataframe= pd.read_json(sentiments_file)

    # Change dialogue acts into numerical variables
    # Let none be 0
    dialogue_rows, dialogue_columns = dialogue_acts_dataframe.shape
    for row in range(dialogue_rows):
        for column in range(dialogue_columns):
            if dialogue_acts_dataframe.iloc[row, column] == 'Emotion':
                dialogue_acts_dataframe.iloc[row, column] = 1
            elif dialogue_acts_dataframe.iloc[row, column] == 'yAnswer':
                dialogue_acts_dataframe.iloc[row, column] = 2
            elif dialogue_acts_dataframe.iloc[row, column] == 'Continuer':
                dialogue_acts_dataframe.iloc[row, column] = 3
            elif dialogue_acts_dataframe.iloc[row, column] == 'whQuestion':
                dialogue_acts_dataframe.iloc[row, column] = 4
            elif dialogue_acts_dataframe.iloc[row, column] == 'System':
                dialogue_acts_dataframe.iloc[row, column] = 5
            elif dialogue_acts_dataframe.iloc[row, column] == 'Accept':
                dialogue_acts_dataframe.iloc[row, column] = 6
            elif dialogue_acts_dataframe.iloc[row, column] == 'Clarify':
                dialogue_acts_dataframe.iloc[row, column] = 7
            elif dialogue_acts_dataframe.iloc[row, column] == 'Emphasis':
                dialogue_acts_dataframe.iloc[row, column] = 8
            elif dialogue_acts_dataframe.iloc[row, column] == 'nAnswer':
                dialogue_acts_dataframe.iloc[row, column] = 9
            elif dialogue_acts_dataframe.iloc[row, column] == 'Greet':
                dialogue_acts_dataframe.iloc[row, column] = 10
            elif dialogue_acts_dataframe.iloc[row, column] == 'Statement':
                dialogue_acts_dataframe.iloc[row, column] = 11
            elif dialogue_acts_dataframe.iloc[row, column] == 'Reject':
                dialogue_acts_dataframe.iloc[row, column] = 12
            elif dialogue_acts_dataframe.iloc[row, column] == 'Bye':
                dialogue_acts_dataframe.iloc[row, column] = 13
            elif dialogue_acts_dataframe.iloc[row, column] == 'Other':
                dialogue_acts_dataframe.iloc[row, column] = 14
            elif dialogue_acts_dataframe.iloc[row, column] == 'ynQuestion':
                dialogue_acts_dataframe.iloc[row, column] = 15
            else:
                dialogue_acts_dataframe.iloc[row, column] = 0
    
    # Let us only consider the compound sentiment of each utterance
    sentiment_rows, sentiment_columns = sentiments_dataframe.shape
    for row in range(sentiment_rows):
        for column in range(sentiment_columns):
            if isinstance(sentiments_dataframe.iloc[row, column], dict):
                sentiments_dataframe.iloc[row, column] = sentiments_dataframe.iloc[row, column]['compound']
            else:
                sentiments_dataframe.iloc[row, column] = 0
    
    # Impute the mean to the missing values NOT DONE YET!!!
    emotion_rows, emotion_columns = emotions_dataframe.shape
    for row in range(emotion_rows):
        for column in range(emotion_columns):
            if emotions_dataframe.iloc[row, column] == None:
                emotions_dataframe.iloc[row, column] = -1
            elif len(emotions_dataframe.iloc[row, column]) == 0:
                emotions_dataframe.iloc[row, column] = -1
            else:
                emotions_dataframe.iloc[row, column] = emotions_dataframe.iloc[row, column][0]

    # Remove constant columns
    # Find constant columns from the dialogue act dataframe
    const_columns = []
    for column in range(dialogue_columns):
        if dialogue_acts_dataframe.iloc[:, column].std() == 0:
            const_columns.append(column)

    # Remove the constant columns from all dataframes
    # Make two copies of the dialogue acts dataframe.
    # From one, the sentiment constant columns will be removed
    # and the emotion constant columns from the other
    dialogue_acts_sentiment_dataframe = dialogue_acts_dataframe.drop(dialogue_acts_dataframe.columns[const_columns], axis=1)
    dialogue_acts_emotion_dataframe = dialogue_acts_dataframe.drop(dialogue_acts_dataframe.columns[const_columns], axis=1)
    sentiments_dataframe = sentiments_dataframe.drop(sentiments_dataframe.columns[const_columns], axis=1)
    emotions_dataframe = emotions_dataframe.drop(emotions_dataframe.columns[const_columns], axis=1)

    # Reset const_columns and set the rows and columns values to the new dimensions, after removing columns
    const_columns = []
    sentiment_rows, sentiment_columns = sentiments_dataframe.shape

    # Find constant columns from the sentiments dataframe
    for column in range(sentiment_columns):
        if sentiments_dataframe.iloc[:, column].std() == 0:
            const_columns.append(column)

    # Remove the sentiment constant columns from sentiment & dialogue acts dataframes
    dialogue_acts_sentiment_dataframe = dialogue_acts_sentiment_dataframe.drop(dialogue_acts_sentiment_dataframe.columns[const_columns], axis=1)
    sentiments_dataframe = sentiments_dataframe.drop(sentiments_dataframe.columns[const_columns], axis=1)

    # Reset const_columns and set the rows and columns values to the new dimensions, after removing columns
    const_columns = []
    emotion_rows, emotion_columns = emotions_dataframe.shape

    # Find constant columns from the emotions dataframe
    for column in range(emotion_columns):
        if emotions_dataframe.iloc[:, column].std() == 0:
            const_columns.append(column)

    # Remove the emotion constant columns from emotion & dialogue acts dataframes
    dialogue_acts_emotion_dataframe = dialogue_acts_emotion_dataframe.drop(dialogue_acts_emotion_dataframe.columns[const_columns], axis=1)
    emotions_dataframe = emotions_dataframe.drop(emotions_dataframe.columns[const_columns], axis=1)

    # Calculate the correlations
    sentiment_correlation = dialogue_acts_sentiment_dataframe.corrwith(sentiments_dataframe)
    emotion_correlation = dialogue_acts_emotion_dataframe.corrwith(emotions_dataframe)

    return emotion_correlation, sentiment_correlation

def main():
    # Prints the matches
    correlations, highest_emotion_correlations = calculate_matching('all_dialogue_predictions.json', 'emos.json', 'sentiments.json')
    print("List of lists with the dialogue acts as the first element and the emotion most associated with that dialogue act as the second element")
    print(highest_emotion_correlations)

    # Prints the correlations
    emotion_correlation, sentiment_correlation = calculate_correlations('all_dialogue_predictions.json', 'emos_upper_level.json', 'sentiments.json')
    print("Dialogue act / emotion correlations")
    print(emotion_correlation)
    print("Dialogue act / sentiment correlations")
    print(sentiment_correlation)

if __name__ == "__main__":
    main()