from sklearn.naive_bayes import MultinomialNB
import numpy as np
from nltk.corpus import stopwords
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from utils import get_dialogs
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import svm

def get_sentiment(utterance):

    analyzer = SentimentIntensityAnalyzer()

    vs = analyzer.polarity_scores(utterance)
    if vs['compound'] <= -0.05:
        return 0
    elif vs['compound'] >= 0.05:
        return 2
    else:
        return 1

def get_pronouns(utterance):

    tokens = word_tokenize(utterance)
    pos_tags = pos_tag(tokens)

    counter = 0
    for pt in pos_tags:
        if pt[1] == 'PRP' or pt[1] == 'PRP$':
            counter += 1

    return counter

def get_negation(utterance):

    negation_terms = ['no', 'not', 'never', 'none', 'nobody', "don't", "can't", 'neither']

    tokens = word_tokenize(utterance)

    counter = 0
    for token in tokens:
        if token in negation_terms:
            counter += 1

    return counter

def get_dialogue_acts(dialogs):
    with open('ijcnlp_dailydialog/dialogues_emotion.txt', 'r') as file:
        acts = file.readlines()
    
    act_tags = {1: 'inform', 2: 'question', 3: 'directive', 4: 'commissive' }

    a = []
    for ac in acts:
        temp = ac.split(' ')
        temp = temp[:-1]
        a.append(temp)
    
    d_acts = []
    i = 0
    for dialog in dialogs:
        if len(dialog) != len(a[i]):
            a[i].append(0)
        for n in a[i]:
            d_acts.append(int(n))

        i += 1

    return d_acts

def get_utterances(dialogs):

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

    return utterances, emotions

def save_features():
    dialogs = get_dialogs()
    utterances, emotions = get_utterances(dialogs)
    acts = get_dialogue_acts(dialogs)

    # Create feature vectors
    features = []
    i = 0
    for utterance in utterances:
        utterance_feature = []
        utterance_feature.append(get_sentiment(utterance))
        utterance_feature.append(get_pronouns(utterance))
        utterance_feature.append(get_negation(utterance))
        utterance_feature.append(acts[i])
        features.append(utterance_feature)
        i += 1
    
    with open('utterance_features.json', 'w') as file:
        json.dump(features, file, indent=4)

def load_dataset():
    dialogs = get_dialogs()
    utterances, emotions = get_utterances(dialogs)

    with open('utterance_features.json', 'r') as file:
        features = json.load(file)

    for i in range(len(emotions)):
        emotions[i] = int(emotions[i])

    counter = 0
    i = 0
    while counter < 65000:
        if emotions[i] == 0:
            emotions.pop(i)
            utterances.pop(i)
            features.pop(i)
            counter += 1
            i -= 1
        i += 1

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(utterances).toarray()
    X = np.hstack([X, features])
    X_train, X_test, y_train, y_test = train_test_split(X, emotions, test_size=0.2)

    return X_train, X_test, y_test, y_train

def multinomialNB():
    X_train, X_test, y_test, y_train = load_dataset()
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    pr = clf.predict(X_test)

    results(y_test, pr, 'Multinomial Naive Bayes')

def randomForest():
    X_train, X_test, y_test, y_train = load_dataset()
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    pr = clf.predict(X_test)

    results(y_test, pr, 'Random forest classifier')

def ridgeClassifier():
    X_train, X_test, y_test, y_train = load_dataset()
    clf = RidgeClassifier(tol=1e-2, solver="sparse_cg")
    clf.fit(X_train, y_train)
    pr = clf.predict(X_test)

    results(y_test, pr, 'Ridge Classifier')

def svm_cl():
    X_train, X_test, y_test, y_train = load_dataset()
    clf = svm.LinearSVC()
    clf.fit(X_train, y_train)
    pr = clf.predict(X_test)

    results(y_test, pr, 'SVM classifier')

def results(y_test, pr, title):
    accuracy = accuracy_score(y_test, pr)
    precision = precision_score(y_test, pr, average='macro')
    recall = recall_score(y_test, pr, average='macro', zero_division=np.nan)

    unique, counts = np.unique(pr, return_counts=True)
    for i in range(len(counts)):
        print('Count ' + str(unique[i]) + ' : ' + str(counts[i]))
    print("Accuracy: " + str(round(accuracy ,3)))
    print("Precision: " + str(round(precision, 3)))
    print("Recall: " + str(round(recall, 3)))

    fig, ax = plt.subplots(figsize=(10, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, pr, ax=ax)
    ax.set_title(title)
    ax.xaxis.set_ticklabels(["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"])
    ax.yaxis.set_ticklabels(["no emotion", "anger", "disgust", "fear", "happiness", "sadness", "surprise"])
    plt.show()

def main():

    #save_features()

    multinomialNB()
    randomForest()
    ridgeClassifier()
    svm_cl()

if __name__ == "__main__":
    main()