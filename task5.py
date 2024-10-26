import json
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import pandas as pd

from utils import get_dialogs
from wnaffect import WNAffect
from emotion import Emotion

wna = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')

def save_upper_level_emotions():
    dialogs = get_dialogs()

    emotions = []
    for dialog in dialogs:
        dialog_emo = []
        for utterance in dialog:
            emo = get_emotions(utterance)
            dialog_emo.append(emo)
        emotions.append(dialog_emo)

    with open("emos_upper_level.json", "w") as f:
        json.dump(emotions, f, indent=4)

def get_emotions(utterance):

    tokens = word_tokenize(utterance)
    pos_tags = pos_tag(tokens)

    emotion_tags = []
    for i in range(len(tokens)):
        emo = wna.get_emotion(tokens[i], pos_tags[i][1])
        if emo != None:
            emotion = get_upper_level_emotion(emo)
            emotion_tags.append(emotion)
            # Emotion.printTree(Emotion.emotions[emo.name])
            # parent = emo.get_level(emo.level - 1)
            # print("parent: " + parent.name)

    return emotion_tags

def get_upper_level_emotion(emo):

    parent = emo.get_level(emo.level - 1)

    while parent.name != "negative-emotion" and parent.name != "positive-emotion" and parent.name != "positive-emotion" and parent.name != "ambiguous-emotion" and parent.name != "neutral-emotion":
        parent = emo.get_level(parent.level - 1)
    
    if parent.name == "ambiguous-emotion" or parent.name == "neutral-emotion":
        return 0
    elif parent.name == "negative-emotion":
        return -1
    else:
        return 1

def compare_and_save():

    with open("emos_upper_level.json", "r") as file:
        upper_level = json.load(file)
    
    with open("sentiments.json", "r") as file:
        sentiments = json.load(file)

    compared_index = []
    for i in range(len(upper_level)):
        dialogs = []
        for j in range(len(upper_level[i])):
            result = get_compared_result(sentiments[i][j]["compound"], upper_level[i][j])
            dialogs.append(result)
        compared_index.append(dialogs)

    save_to_excel(compared_index, upper_level, sentiments, get_dialogs())
    
def save_to_excel(compared_index, emotion_values, sentiments, dialogs):
    data = {}
    compability_index_list = []
    emotion_values_list = []
    sentiments_list = []
    utterances_list = []
    for i in range(len(compared_index)):
        compability_index_list.extend(compared_index[i])
        for j in range(len(compared_index[i])):
            if len(emotion_values[i][j]) == 0:
                emotion_values_list.append("None")
            else:
                emotion_values_list.append(emotion_values[i][j])
            sentiments_list.append(sentiments[i][j]["compound"])
            utterances_list.append(dialogs[i][j])
    data["compability index"] = compability_index_list
    data["emotion value"] = emotion_values_list
    data["sentiment"] = sentiments_list
    data["utterance"] = utterances_list

    df = pd.DataFrame(data)
    df.to_excel("task5_data.xlsx", index=False)

def get_compared_result(sentiment_value, emotion_values):
    if sentiment_value >= 0.05:
        if 1 in emotion_values:
            if 0 not in emotion_values and -1 not in emotion_values:
                return 1
            return 0.5
        else:
            return 0
    elif sentiment_value <= -0.05:
        if -1 in emotion_values:
            if 1 not in emotion_values and 0 not in emotion_values:
                return 1
            return 0.5
        else:
            return 0
    else:
        if 0 in emotion_values:
            if 1 not in emotion_values and -1 not in emotion_values:
                return 1
            return 0.5
        else:
            return 0
def main():
    
    #save_upper_level_emotions()
    compare_and_save()

    return

if __name__ == "__main__":
    main()