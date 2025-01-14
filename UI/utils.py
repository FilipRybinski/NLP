import os
import pandas as pd
from joblib import load
from constants.constants import DICTIONARY

def predict_review(classifier, vectorizer, review):
    try:
        review_vector = vectorizer.transform([review])
        prediction = classifier.predict(review_vector)
        return DICTIONARY.POSITIVE if prediction[0] == 1 else DICTIONARY.NEGATIVE
    except Exception as e:
        return None

def load_model(classifier_file, vectorizer_file):
    try:
        classifier = load(classifier_file)
        vectorizer = load(vectorizer_file)
        return classifier, vectorizer
    except Exception as e:
        return None, None

def load_csv_data():
    csv_path = os.path.join(DICTIONARY.MODELS_RANKING_PATH, f"{DICTIONARY.CLASSIFICATION_RESULT_FILE}.{DICTIONARY.CSV_EXTENSION}")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path).round(4)
    return None

def get_best_csv_classifier_vectorizer():
    csv_data = load_csv_data()
    if csv_data is None:
        return None, None, None, None
    if "Vectorizer" not in csv_data.columns or "Classifier" not in csv_data.columns:
        return None, None, None, None
    first_row = csv_data.iloc[0]
    vectorizer_name = first_row["Vectorizer"] or None
    classifier_name = first_row["Classifier"] or None
    if vectorizer_name is None or classifier_name is None:
        return None, None, None, None
    vectorizer_file = os.path.join(DICTIONARY.MODELS_PATH, DICTIONARY.VECTORIZER_PATH, f"{vectorizer_name}_{DICTIONARY.VECTORIZER_FILE}.{DICTIONARY.JOBLIB_EXTENSION}")
    classifier_file = os.path.join(DICTIONARY.MODELS_PATH, f"{classifier_name}_{vectorizer_name}_{DICTIONARY.MODEL_FILE}.{DICTIONARY.JOBLIB_EXTENSION}")
    classifier, vectorizer = load_model(classifier_file, vectorizer_file)
    return classifier_name, vectorizer_name, classifier, vectorizer
