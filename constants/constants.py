from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC



class DICTIONARY:
    VECTORIZER = "Vectorizer"
    CLASSIFIER = "Classifier"
    TRAIN_ACCURACY = "Train Accuracy"
    TEST_ACCURACY = "Test Accuracy"
    TEST_F1_SCORE = "Test F1 Score"
    DATA_PATH = "../data"
    TRAIN_PATH = "train"
    TEST_PATH = "test"
    CLASSIFICATION_RESULT_FILE = "classification_results.csv"
    MODELS_RANKING_PATH = "ranking"
    MODEL_FILE="model"
    VECTORIZER_FILE="vectorizer"
    MODELS_PATH= "models"
    VECTORIZER_PATH = "vectorizer"
    JOBLIB_EXTENSION = ".joblib"
    POSITIVE = "Positive"
    NEGATIVE= "Negative"

# Wektoryzacja
VECTORIZERS = {
    "CountVectorizer": CountVectorizer(),
    "TfidfVectorizer": TfidfVectorizer(),
}

# Klasyfikatory
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(max_iter=1000),
    "SVM": SVC(),
    "NaiveBayes": MultinomialNB(),
}