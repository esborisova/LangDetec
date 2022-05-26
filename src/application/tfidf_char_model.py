"""Pipeline for building language detection model based on character n-grams"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import ConfusionMatrixDisplay


df = pd.read_pickle("../../pkl/dataset.pkl")
df["language"].value_counts()

features = df["text"]

le = LabelEncoder()
y = le.fit(df["language"])
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
y = y.transform(df["language"])

x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.20)

tfidf_vectorizer = TfidfVectorizer(lowercase=False, analyzer="char", ngram_range=(1, 2))
X_train = tfidf_vectorizer.fit_transform(x_train)

X_test = tfidf_vectorizer.transform(x_test)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

matrix = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(12, 10))
c_mat = ConfusionMatrixDisplay(matrix, display_labels=["da", "de", "en", "sv"])
c_mat.plot(ax=ax, cmap=plt.cm.Blues)
plt.savefig("../../figs/tfidf_char_c_matrix.pdf")

target_names = ["da", "de", "en", "sv"]
report = classification_report(
    y_test, y_pred, target_names=target_names, output_dict=True
)
df = pd.DataFrame(report).transpose()
df.to_csv("../../csv/tfidf_char_class_report.csv", index=True)
