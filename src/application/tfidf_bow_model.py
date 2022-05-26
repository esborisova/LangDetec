"""Pipeline for building language detection model based on BOW"""
import sys

sys.path.append("../langdetect/")
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from preprocessing_text import identity_tokenizer


df = pd.read_pickle("../../pkl/tokens.pkl")

features = df["tokens"]

le = LabelEncoder()
y = le.fit(df["language"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
y = y.transform(df["language"])

x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.20)

tfidf_vectorizer = TfidfVectorizer(
    lowercase=False, tokenizer=identity_tokenizer, ngram_range=(1, 1)
)
X_train = tfidf_vectorizer.fit_transform(x_train)

X_test = tfidf_vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)
print(matrix)

target_names = ["da", "de", "en", "sv"]
print(classification_report(y_test, y_pred, target_names=target_names))
