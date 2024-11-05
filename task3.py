# NLP project 16
# 26.10.2024
# Sami Karhumaa

from wnaffect import WNAffect
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
import nltk
from emotion import Emotion

from utils import get_dialogs

wna = WNAffect('wordnet-1.6/', 'wn-domains-3.2/')

# Tokenizes each utterance and tags each word’s part of speech (POS).
# Queries WNAffect to get emotions for each word in the utterance based on POS tags, accumulating any detected emotions in a list.
# Returns list of emotions for utterance
def get_emotions(utterance):

    tokens = word_tokenize(utterance)
    pos_tags = pos_tag(tokens)

    emotions = []
    for i in range(len(tokens)):
        emo = wna.get_emotion(tokens[i], pos_tags[i][1])
        if emo != None:
            emotions.append(emo.name)
            Emotion.printTree(Emotion.emotions[emo.name])
            parent = emo.get_level(emo.level - 1)
            print("parent: " + parent.name)

    return emotions

# Compares the predicted emotions against expected labels from dialogues_emotion.txt.
# Calculates accuracy and precision metrics by comparing the predicted emotions (from emos.json) with the labels.
#   Consider as match if at least one emos.json label matches with tag from dialogues_emotion.txt.
# validate uses a confusion matrix approach with:
#   True Positives (TP), True Negatives (TN), False Positives (FP), and False Negatives (FN).
# Prints accuracy and precision scores.
def validate(emotions):

    with open("dialogues_emotion.txt", "r", encoding="utf-8") as file:
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

# Loads emotions from emos.json and validates them against the expected labels in dialogues_emotion.txt.
def main():

    with open("emos.json", "r") as f:
        data = json.load(f)

    validate(data)

    return


if __name__ == "__main__":

    # Downloads NLTK resources required for POS tagging.
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('tagsets')
    # nltk.download('tagsets_json')
    # nltk.help.upenn_tagset()

    # process dialogues and store emotions
    # save_emotions()

    main()