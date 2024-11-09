# NLP project 16
# 26.10.2024
# Sami Karhumaa

from wnaffect import WNAffect
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
import nltk
from emotion import Emotion
from sklearn.metrics import accuracy_score, precision_score, recall_score
import random
import numpy as np

wna = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')

def get_dialogs():

    with open("ijcnlp_dailydialog/dialogues_text.txt", "r", encoding="utf-8") as file:
        dialogs = file.readlines()

    parsed_dialogs = []
    for dialog in dialogs:
        d = dialog.split("__eou__")
        d = d[:-1]
        parsed_dialogs.append(d)

    return parsed_dialogs

def get_emotions(dialogs):
    with open("ijcnlp_dailydialog/dialogues_emotion.txt", "r", encoding="utf-8") as file:
        emotion_numbers = file.readlines()

    en = []
    for e in emotion_numbers:
        a = e.split(" ")
        a = a[:-1]
        en.append(a)

    utterances = []
    emotions = []
    i = 0
    for dialog in dialogs:
        if len(dialog) != len(en[i]):
            en[i].append(0)
        utterances.extend(dialog)
        emotions.extend(en[i])

        i += 1

    return emotions

def get_emotions(dialogs):
    with open("ijcnlp_dailydialog/dialogues_emotion.txt", "r", encoding="utf-8") as file:
        emotion_numbers = file.readlines()

    en = []
    for e in emotion_numbers:
        a = e.split(" ")
        a = a[:-1]
        en.append(a)

    utterances = []
    emotions = []
    i = 0
    for dialog in dialogs:
        if len(dialog) != len(en[i]):
            en[i].append(0)
        utterances.extend(dialog)
        emotions.extend(en[i])

        i += 1

    return emotions

# Compares the predicted emotions against expected labels from dialogues_emotion.txt.
# Calculates accuracy and precision metrics by comparing the predicted emotions (from emos.json) with the labels.
#   Consider as match if at least one emos.json label matches with tag from dialogues_emotion.txt.
# validate uses a confusion matrix approach with:
#   True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
# Prints accuracy and precision scores.
def validate(emotions):

    with open("ijcnlp_dailydialog/dialogues_emotion.txt", "r", encoding="utf-8") as file:
        emotion_numbers = file.readlines()

    en = []
    for e in emotion_numbers:
        a = e.split(" ")
        a = a[:-1]
        en.append(a)

    emo_tags = {0: "no emotion", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    no_emos = 0
    no_predictions =  0

    for i in range(len(emotions)):
        for j in range(len(emotions[i])):
            try:
                tag = emo_tags[int(en[i][j])]
                utterance_emo = emotions[i][j]
                if len(utterance_emo) == 0:
                    no_predictions += 1
                if tag in utterance_emo:
                    tp += 1
                elif tag == "no emotion":
                    no_emos += 1
                    if len(utterance_emo) == 0:
                        tn += 1
                    else:
                        fp += 1
                elif len(utterance_emo) == 0:
                    fn += 1
                else:
                    fp += 1
            except IndexError:
                print("index error: " + str(i) + " " + str(j))

    total = fp + fn + tp + tn

    accuracy = (tp + tn) / total
    precision = tp / (tp + fp)

    ratio_no_emos = no_emos / total
    ratio_no_predictions = no_predictions / total
    
    print("WNAffect scores: ")
    print("Total utterance count: " + str(total))
    print("Accuracy: " + str(round(accuracy ,3)))
    print("Precision: " + str(round(precision, 3)))
    print("No tag ratio: " + str(round(ratio_no_emos, 3)))
    print("No prediction ratio: " + str(round(ratio_no_predictions, 3)))

    return

def validate_m(emotions):
    
    emotion_tags = get_emotions(get_dialogs())

    emo_tags = {0: "no emotion", 1: "anger", 2: "disgust", 3: "fear", 4: "happiness", 5: "sadness", 6: "surprise"}

    y_true = []
    for tag in emotion_tags:
        y_true.append(emo_tags[int(tag)])

    y_pred = get_pred_class(emotions, y_true)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro', zero_division=np.nan)

    print("Accuracy: " + str(round(accuracy ,3)))
    print("Precision: " + str(round(precision, 3)))
    print("Recall: " + str(round(recall, 3)))

def get_pred_class(emotions, y_true):

    y_pred = []
    for i in range(len(emotions)):
        if y_true[i] in emotions[i]:
            y_pred.append(y_true[i])
        elif len(emotions[i]) == 0:
            y_pred.append('no emotion')
        elif 'negative-fear' in emotions[i] or 'ambiguous-fear' in emotions[i]:
            if y_true[i] == 'fear':
                y_pred.append('fear')
            else:
                y_pred.append(emotions[i][0])
        else:
            y_pred.append(emotions[i][0])
    
    return y_pred

# Calls get_dialogs to load dialogues.
# Saves emotions fro each utterance to emos.json
def save_emotions():
    dialogs = get_dialogs()

    emotions = []

    for dialog in dialogs:
        dialog_emo = []
        for utterance in dialog:
            emo = get_emotions(utterance)
            dialog_emo.append(emo)
        emotions.append(dialog_emo)

    with open("emos.json", "w") as f:
        json.dump(emotions, f, indent=4)

def results_a1():
    with open("emos.json", "r") as f:
        data = json.load(f)
    emos = []
    for d in data:
        for u in d:
            emos.append(u)

    validate_m(emos)

def results_a2():
    with open("emos_upperlevel.json", "r") as f:
        data = json.load(f)
    emos = []
    for d in data:
        for u in d:
            emos.append(u)

    validate_m(emos)

# Loads emotions from emos.json and validates them against the expected labels in dialogues_emotion.txt.
def main():

    print("Results for A1")
    results_a1()
    print("")
    print("Results for A2")
    results_a2()

if __name__ == "__main__":

    # Downloads NLTK resources required for POS tagging.
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('tagsets')
    # nltk.download('tagsets_json')
    # nltk.help.upenn_tagset()

    # process dialogues and store emotions
    # save_emotions()

    main()