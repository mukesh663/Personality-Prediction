from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    result = "Not Found"
    if request.method == 'POST':
        result = predict(request.get_json())
        return jsonify(result)

######################################################### Model ############################################################

# importing dependencies here
import numpy as np
import pandas as pd
import os

import re
import nltk

# lemmatizing
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# sentiment scoring
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# pos tagging
from nltk.tokenize import word_tokenize

# accuracy scores
from sklearn.metrics import (
    f1_score,
    accuracy_score,
)

# performance check
import time

# sparse to dense
from sklearn.base import TransformerMixin


class DenseTransformer(TransformerMixin):
    def fit(self, X, y=None, **fit_params):
        return self

    def transform(self, X, y=None, **fit_params):
        return X.todense()


# importing model
from joblib import load

mbti = [
    "INFP",
    "INFJ",
    "INTP",
    "INTJ",
    "ENTP",
    "enfp",
    "ISTP",
    "ISFP",
    "ENTJ",
    "ISTJ",
    "ENFJ",
    "ISFJ",
    "ESTP",
    "ESFP",
    "ESFJ",
    "ESTJ",
]

def unique_words(s):
    unique = set(s.split(" "))
    return len(unique)

def emojis(post):
    emoji_count = 0
    words = post.split()
    for e in words:
        if "http" not in e:
            if e.count(":") == 2:
                emoji_count += 1
    return emoji_count

def colons(post):
    colon_count = 0
    words = post.split()
    for e in words:
        if "http" not in e:
            colon_count += e.count(":")
    return colon_count

def lemmitize(s):
    lemmatizer = WordNetLemmatizer()
    new_s = ""
    for word in s.split(" "):
        lemmatizer.lemmatize(word)
        if word not in stopwords.words("english"):
            new_s += word + " "
    return new_s[:-1]

def clean(s):
    # remove urls
    s = re.sub(re.compile(r"https?:\/\/(www)?.?([A-Za-z_0-9-]+).*"), "", s)
    # remove emails
    s = re.sub(re.compile(r"\S+@\S+"), "", s)
    # remove punctuation
    s = re.sub(re.compile(r"[^a-z\s]"), "", s)
    # Make everything lowercase
    s = s.lower()
    # remove all personality types
    for type_word in mbti:
        s = s.replace(type_word.lower(), "")
    
    return s


def prep_counts(s):
    clean_s = clean(s)
    d = {
        "clean_posts": lemmitize(clean_s),
        "link_count": s.count("http"),
        "youtube": s.count("youtube") + s.count("youtu.be"),
        "img_count": len(re.findall(r"(\.jpg)|(\.jpeg)|(\.gif)|(\.png)", s)),
        "upper": len([x for x in s.split() if x.isupper()]),
        "char_count": len(s),
        "word_count": clean_s.count(" ") + 1,
        "qm": s.count("?"),
        "em": s.count("!"),
        "colons": colons(s),
        "emojis": emojis(s),
        "unique_words": unique_words(clean_s),
        "ellipses": len(re.findall(r"\.\.\.\ ", s)),
    }
    return clean_s, d


def prep_sentiment(s):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(s)
    d = {
        "compound_sentiment": score["compound"],
        "pos_sentiment": score["pos"],
        "neg_sentiment": score["neg"],
        "neu_sentiment": score["neu"],
    }
    return d


def tag_pos(s):
    tagged_words = nltk.pos_tag(word_tokenize(s))
    tags_dict = {
        "ADJ_avg": ["JJ", "JJR", "JJS"],
        "ADP_avg": ["EX", "TO"],
        "ADV_avg": ["RB", "RBR", "RBS", "WRB"],
        "CONJ_avg": ["CC", "IN"],
        "DET_avg": ["DT", "PDT", "WDT"],
        "NOUN_avg": ["NN", "NNS", "NNP", "NNPS"],
        "NUM_avg": ["CD"],
        "PRT_avg": ["RP"],
        "PRON_avg": ["PRP", "PRP$", "WP", "WP$"],
        "VERB_avg": ["MD", "VB", "VBD", "VBG", "VBN", "VBP", "VBZ"],
        ".": ["#", "$", "''", "(", ")", ",", ".", ":"],
        "X": ["FW", "LS", "UH"],
    }
    d = dict.fromkeys(tags_dict, 0)
    for tup in tagged_words:
        tag = tup[1]
        for key, val in tags_dict.items():
            if tag in val:
                tag = key
        d[tag] += 1
    return d


def prep_data(s):
    clean_s, d = prep_counts(s)
    sentiment = prep_sentiment(lemmitize(clean_s))
    d.update(sentiment)
    d.update(tag_pos(clean_s))
    features = [
        "clean_posts",
        "compound_sentiment",
        "ADJ_avg",
        "ADP_avg",
        "ADV_avg",
        "CONJ_avg",
        "DET_avg",
        "NOUN_avg",
        "NUM_avg",
        "PRT_avg",
        "PRON_avg",
        "VERB_avg",
        "qm",
        "em",
        "colons",
        "emojis",
        "word_count",
        "unique_words",
        "upper",
        "link_count",
        "ellipses",
        "img_count",
    ]

    return pd.DataFrame([d])[features], sentiment

def trace_back(combined):
    
    type_list = [
    {"0": "E", "1": "I"},
    {"0": "S", "1": "N"},
    {"0": "F", "1": "T"},
    {"0": "P", "1": "J"},
    ]

    type_dict = {"I":"Introversion", "E":"Extroversion", "S": "Sensing", "N":"Intuition",
                 "F":"Feeling", "T":"Thinking", "P":"Perceiving", "J":"Judging"}
    result = []
    for num in combined:
        s = ""
        lt = []
        for i in range(len(num)):
            res = type_list[i][num[i]]
            lt.append(type_dict[res])
            s += res
        result.append(s)
        result.append(lt)
        
    return result
    
def combine_classes(y_pred1, y_pred2, y_pred3, y_pred4):

    combined = []
    for i in range(len(y_pred1)):
        combined.append(str(y_pred1[i]) + str(y_pred2[i]) + str(y_pred3[i]) + str(y_pred4[i]))
    
    result = trace_back(combined)
    return result


def predict(s):

    s = s['text']
    if s == "":
        return "No text" 

    X, output = prep_data(s)

    # loading the 4 models
    EorI_model = load("app/trained_weights/Introversion.joblib")
    SorN_model = load("app/trained_weights/Intuition.joblib")
    TorF_model = load("app/trained_weights/Thinking.joblib")
    JorP_model = load("app/trained_weights/Judging.joblib")

    # predicting
    EorI_pred = EorI_model.predict(X)
    SorN_pred = SorN_model.predict(X)
    TorF_pred = TorF_model.predict(X)
    JorP_pred = JorP_model.predict(X)

    # combining the predictions from the 4 models
    result = combine_classes(EorI_pred, SorN_pred, TorF_pred, JorP_pred)

    for key in output.keys():
        output[key]= round(output[key]*100, 2)
    output["result"] = result[0]
    output["typeExp"] = " - ".join(result[1])
    
    return output

######################################################################################################################################