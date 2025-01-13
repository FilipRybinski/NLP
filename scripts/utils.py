import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from concurrent.futures import ThreadPoolExecutor
from constants.constants import DICTIONARY

# Wektoryzacja
VECTORIZERS = {
    "CountVectorizer": CountVectorizer(
        max_features=20000,  # Użycie ograniczonej liczby cech
        ngram_range=(1, 2),  # Unigramy i bigramy
        stop_words="english",  # Usuwanie stop words
        binary=True  # Wartości binarne dla lepszej klasyfikacji sentymentu
    ),
    "TfidfVectorizer": TfidfVectorizer(
        max_features=20000,  # Użycie ograniczonej liczby cech
        ngram_range=(1, 2),  # Unigramy i bigramy
        stop_words="english",  # Usuwanie stop words
        sublinear_tf=True  # Skalowanie TF w trybie sublinearnym
    ),
    "HashingVectorizer": HashingVectorizer(
        n_features=20000,  # Większa liczba cech dla lepszego modelu
        alternate_sign=False,  # Wyłączenie naprzemiennych znaków, aby dopasować się do modeli, które oczekują tylko dodatnich wartości
        ngram_range=(1, 2)  # Unigramy i bigramy
    ),
}

# Klasyfikatory
CLASSIFIERS = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000,  # Większa liczba iteracji, aby zapewnić konwergencję
        solver="liblinear",  # Solver dobry dla małych i średnich zbiorów danych
        C=1.0  # Współczynnik regularyzacji (1.0 to domyślna wartość)
    ),
    "SVM": SVC(
        kernel="linear",  # Kernel liniowy sprawdza się dobrze przy tekstach
        C=1.0  # Współczynnik regularyzacji
    ),
    "NaiveBayes": MultinomialNB(
        alpha=0.1  # Lekka regularyzacja Laplace'a dla uniknięcia problemów z rzadkimi słowami
    ),
}

def process_model(vectorizer, vec_name, train_reviews, test_reviews, train_labels, test_labels, rootDir):
    train, test = transform_data(vectorizer, train_reviews, test_reviews)
    vectorizer_filename = adjust_path(
        os.path.join(DICTIONARY.MODELS_PATH, DICTIONARY.VECTORIZER_PATH, f"{vec_name}_{DICTIONARY.VECTORIZER_FILE}.{DICTIONARY.JOBLIB_EXTENSION}"),
        condition=rootDir,
    )
    save_vectorizer(vectorizer, vectorizer_filename)
    results = []

    for clf_name, classifier in CLASSIFIERS.items():
        model_filename = adjust_path(
            os.path.join(DICTIONARY.MODELS_PATH, f"{clf_name}_{vec_name}_{DICTIONARY.MODEL_FILE}.{DICTIONARY.JOBLIB_EXTENSION}"),
            condition=rootDir,
        )
        classifier.fit(train, train_labels)
        save_model(classifier, model_filename)

        train_pred = classifier.predict(train)
        test_pred = classifier.predict(test)

        results.extend(rate_model(train_pred, test_pred, vec_name, clf_name, train_labels, test_labels))
    return results

def create_models(rootDir):
    train_reviews, train_labels = load_data(adjust_path(os.path.join(DICTIONARY.DATA_PATH, DICTIONARY.TRAIN_PATH), condition=rootDir))
    test_reviews, test_labels = load_data(adjust_path(os.path.join(DICTIONARY.DATA_PATH, DICTIONARY.TEST_PATH), condition=rootDir))
    make_model_dir(rootDir)

    all_results = []

    with ThreadPoolExecutor() as executor:
        futures = [
            executor.submit(process_model, vectorizer, vec_name, train_reviews, test_reviews, train_labels, test_labels, rootDir)
            for vec_name, vectorizer in VECTORIZERS.items()
        ]

        for future in futures:
            all_results.extend(future.result())

    save_rate_model(all_results, rootDir)


def load_data(data_dir):
    reviews, labels = [], []
    for label in ["pos", "neg"]:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            with open(file_path, encoding="utf-8") as f:
                reviews.append(f.read())
                labels.append(1 if label == "pos" else 0)
    return reviews, labels


def make_model_dir(rootDir):
    os.makedirs(adjust_path(f"{DICTIONARY.MODELS_PATH}/{DICTIONARY.VECTORIZER_PATH}", condition=rootDir), exist_ok=True)


def transform_data(vectorizer, train_reviews, test_reviews):
    return vectorizer.fit_transform(train_reviews), vectorizer.transform(test_reviews)


def save_vectorizer(vectorizer, vectorizer_filename):
    try:
        dump(vectorizer, vectorizer_filename)
    except Exception as e:
        print(f"Error saving vectorizer: {e}")


def save_model(classifier, model_filename):
    try:
        dump(classifier, model_filename)
    except Exception as e:
        print(f"Error saving model: {e}")


def rate_model(train_pred, test_pred, vec_name, clf_name, train_labels, test_labels):
    try:
        train_accuracy = accuracy_score(train_labels, train_pred)
        test_accuracy = accuracy_score(test_labels, test_pred)
        test_f1 = f1_score(test_labels, test_pred)

        results = [{
            DICTIONARY.VECTORIZER: vec_name,
            DICTIONARY.CLASSIFIER: clf_name,
            DICTIONARY.TRAIN_ACCURACY: train_accuracy,
            DICTIONARY.TEST_ACCURACY: test_accuracy,
            DICTIONARY.TEST_F1_SCORE: test_f1,
        }]
        return results
    except Exception as e:
        print(f"Error rating model: {e}")
        return []


def save_rate_model(results, rootDir):
    try:
        results_df = pd.DataFrame(results)
        results_df.sort_values(by=DICTIONARY.TEST_ACCURACY, ascending=False, inplace=True)
        output_path = adjust_path(os.path.join(DICTIONARY.MODELS_RANKING_PATH, f"{DICTIONARY.CLASSIFICATION_RESULT_FILE}.{DICTIONARY.CSV_EXTENSION}"), condition=rootDir)
        results_df.to_csv(output_path, index=False)
    except Exception as e:
        print(f"Error saving model results: {e}")


def adjust_path(base_path, condition=True):
    return os.path.join("..", base_path) if condition else base_path
