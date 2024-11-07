# NLP project 16
# 3.11.2024
# Sami Karhumaa

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from utils import get_dialogs
import numpy as np
import matplotlib.pyplot as plt

def main():
    
    analyzer = SentimentIntensityAnalyzer()
    dialogs = get_dialogs()

    dialog_sentiments = []
    for dialog in dialogs:
        utterances = []
        for utterance in dialog:
            vs = analyzer.polarity_scores(utterance)
            utterances.append(vs)
            print("{:-<65} {}".format(utterance, str(vs)))
        dialog_sentiments.append(utterances)

    with open("sentiments.json", "w", encoding="utf-8") as file:
        json.dump(dialog_sentiments, file, indent=4)

    return

def results():

    with open("sentiments.json", 'r') as file:
        sentiments = json.load(file)
    
    compounds = []
    for dialog in sentiments:
        for sentiment in dialog:
            compounds.append(sentiment['compound'])

    bins = np.linspace(-1,1)
    data = compounds
    plt.hist(data, bins=bins, edgecolor='black')
    plt.xticks(rotation=90)
    plt.title('Vader sentiment analysis')
    plt.xlabel('Compound score')

    plt.show()

if __name__ == "__main__":
    #main()
    results()