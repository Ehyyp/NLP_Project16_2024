from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import json
from utils import get_dialogs


def main():
    
    analyzer = SentimentIntensityAnalyzer()
    dialogs = get_dialogs()

    #print(analyzer.polarity_scores("Isn’t he the best instructor? I think he’s so hot. Wow! I really feel energized, dont’t you?"))

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

if __name__ == "__main__":
    main()