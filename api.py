"""Simple NLP app"""
import re
import warnings
import pickle
import nltk
import pandas as pd
import numpy as np
from typing import List
from io import BytesIO
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
from fastapi import FastAPI, Response, File
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

stop_words = stopwords.words('english')

for dependency in (
    "brown",
    "names",
    "wordnet",
    "averaged_perceptron_tagger",
    "universal_tagset"
):
    nltk.download(dependency)
warnings.filterwarnings("ignore")
nltk.download('stopwords')
nltk.download('wordnet')


# Create app instance
app = FastAPI(
    title="Simple NLP app - movie prediction",
    description="Implementation of two endpoints with FastAPI. The user can POST train.csv to the training endpoint, with which the model is created and trained, and the test.csv file to the predict endpoint, to predict the top 5 movie genres.",
    version="1.0"
)


def lemmatize(text: List[str]) -> List[str]:
    """Lematize."""
    lemmatizer = WordNetLemmatizer()
    output = []
    for word in text:
        output.append(lemmatizer.lemmatize(word))
    return output


def stemming(text: List[str]) -> List[str]:
    """Stem words."""
    stemmer = PorterStemmer()
    output = []
    for word in text:
        output.append(stemmer.stem(word))
    return output


def preprocess_text(text: str) -> str:
    """Preprocess text."""
    text = text.lower()
    text = re.sub(r"[^A-Za-z0-9]", " ", text)  # Remove punctuation
    text = re.sub(r"\'s", " ", text)  # Remove apostrophe
    text = re.sub(r'http\S+', ' link ', text)  # Remove links
    text = re.sub(r'\b\d+(?:\.\d+)?\s+', '', text)  # remove numbers
    out = text.split()
    out = [w for w in out if w not in stop_words]   # Remove stopwords
    out_ = lemmatize(out)
    out_ = stemming(out_)
    text = " ".join(out_)
    return text


def split_categories(text: str) -> List[str]:
    """Split the genres string into an array of individual genres."""
    out = text.split()
    return out


@app.post("/train")
def train(file: bytes = File(...)) -> None:
    """Train a predictive model to rank movie genres based on their synopsis."""
    # Get file from endpoint
    df = pd.read_csv(BytesIO(file))

    # Preprocess the synopsis
    df["clean"] = df["synopsis"].apply(preprocess_text)
    df["split"] = df["genres"].apply(split_categories)

    # Convert the labels into categorical vectors
    multilabel_binarizer = MultiLabelBinarizer()
    multilabel_binarizer.fit(df['split'])

    #  transform target variable
    y = multilabel_binarizer.transform(df['split'])
    x = df["clean"]

    # Vectorize the "cleaned" text with the TF-IDF (Term Frequency — Inverse Document Frequency)
    tfidf_vectorizer = TfidfVectorizer(max_features=10000, stop_words=stop_words)
    xtrain_tfidf = tfidf_vectorizer.fit_transform(np.array(x))

    # Create a logistic regression model and the One-VS-Rest Classifier and fit the training data
    lr = LogisticRegression(random_state=23, solver='saga', max_iter=3000, C=1.5, n_jobs=-1, verbose=True)
    clf = OneVsRestClassifier(lr, n_jobs=-1)
    clf.fit(xtrain_tfidf.toarray(), y)

    # Pickle what is needed for the predict endpoint as well
    pickle.dump(clf, open("classifier.p", "wb"))
    pickle.dump(multilabel_binarizer, open("binarizer.p", "wb"))
    pickle.dump(tfidf_vectorizer, open("tfidf.p", "wb"))


@app.post("/predict")
def predict(file: bytes = File(...)) -> None:
    """Predict the genres of the movies in the test set."""
    df = pd.read_csv(BytesIO(file))

    # Re-load what is needed from the train endpoint
    clf_ = pickle.load(open("classifier.p", "rb"))
    multilabel_binarizer = pickle.load(open("binarizer.p", "rb"))
    tfidf_vectorizer = pickle.load(open("tfidf.p", "rb"))

    # Get the different classes names
    categories = multilabel_binarizer.classes_

    # Pre-process the test synopsis texts and vectorize with TF-IDF as well
    df["clean"] = df["synopsis"].apply(preprocess_text)
    x = df["clean"]
    xtest_tfidf = tfidf_vectorizer.transform(np.array(x))

    # Generate predictions with the loaded classifier
    ypred = clf_.predict_proba(xtest_tfidf.toarray())  # Predict probabilities
    y_new = []
    for i in ypred:
        sorted_ = i.argsort()[-5:][::-1]  # Get the indices of the 5 categories with maximal probabilities
        out = np.array([categories[j] for j in sorted_])  # Get top 5 category names
        y_new.append(' '.join(out))  # Join all as a string again
        y_ = pd.Series(y_new)  # Turn into panas series to append to dataFrame.

    df["genres"] = y_.astype(object)  # Append predictions to the test DataFrame
    df = df.drop(columns=["clean", "year", "synopsis"])  # Delete columns that are not needed (based on the description at GitLab)
    output = df.to_csv('submission.csv', mode='w', index=False)  # Save into a new csv file
    return Response(content=output)
