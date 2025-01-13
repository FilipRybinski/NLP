import pandas as pd
from PyQt5.QtWidgets import QMessageBox

from constants.constants import DICTIONARY
from utils.utils import load_model


def load_ranking_data(self, csv_path):
    try:
        data = pd.read_csv(csv_path)
        return data.round(4)
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Failed to load CSV: {e}")


def get_best_model_and_vectorizer(self):
    first_row = self.data_ranking.iloc[0]
    vectorizer_name = first_row["Vectorizer"]
    classifier_name = first_row["Classifier"]

    self.classifier_name = classifier_name
    self.vectorizer_name = vectorizer_name

    vectorizer_file = f"{DICTIONARY.MODELS_PATH}/{DICTIONARY.VECTORIZER_PATH}/{vectorizer_name}_{DICTIONARY.VECTORIZER_FILE}{DICTIONARY.JOBLIB_EXTENSION}"
    model_file = f"{DICTIONARY.MODELS_PATH}/{classifier_name}_{vectorizer_name}_{DICTIONARY.MODEL_FILE}{DICTIONARY.JOBLIB_EXTENSION}"

    load_model(self, model_file, vectorizer_file )