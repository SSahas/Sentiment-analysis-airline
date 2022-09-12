import joblib
from fastapi import FastAPI
import pandas as pd
import string
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
data = pd.read_csv("text.csv")


def clean_text(text):
    text = "".join([word.lower()
                   for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    return text


vectoriz = TfidfVectorizer(analyzer=clean_text)
vectorizer = vectoriz.fit(data["text"])


def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count/(len(text) - text.count(" ")), 3)*100


app = FastAPI()

model = joblib.load("randomforestmodel.pkl")


@app.get("/")
def read_root():
    return {"Hello, this is the sentiment analysis classifier given as a assigment by Truefoundry"}


@app.get("/Prediction")
def get_prediction(review: str):
    text = str(review)
    trans = vectorizer.transform([text])
    body_len = len(text) - text.count(" ")
    punct = count_punct(text)
    k = {"body_len": [body_len], "punc%": [punct]}
    df = pd.DataFrame(k)
    test_vect = pd.concat([df.reset_index(drop=True),
                           pd.DataFrame(trans.toarray())], axis=1)
    prediction = model.predict(test_vect)
    return {"text": text, "Prediction": prediction[0]}
