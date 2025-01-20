import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, HashingVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from joblib import dump
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from concurrent.futures import ThreadPoolExecutor
from constants.constants import DICTIONARY

VECTORIZERS = {
    "CountVectorizer_Ngram11": CountVectorizer(
        max_features=300,
        ngram_range=(1, 1),
        stop_words="english"
    ),
    "CountVectorizer_Ngram12": CountVectorizer(
        max_features=600,
        ngram_range=(1, 2),
        stop_words="english"
    ),
    "CountVectorizer_Ngram21": CountVectorizer(
        max_features=900,
        ngram_range=(2, 1),
        stop_words=None
    ),
    "TfidfVectorizer_Ngram12": TfidfVectorizer(
        max_features=300,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True
    ),
    "TfidfVectorizer_Ngram22": TfidfVectorizer(
        max_features=600,
        ngram_range=(2, 2),
        stop_words="english",
        sublinear_tf=False
    ),
    "TfidfVectorizer_Ngram11": TfidfVectorizer(
        max_features=450,
        ngram_range=(1, 1),
        stop_words=None,
        sublinear_tf=True
    ),
    "HashingVectorizer_Ngram11": HashingVectorizer(
        n_features=300,
        alternate_sign=False,
        ngram_range=(1, 1)
    ),
    "HashingVectorizer_Ngram12": HashingVectorizer(
        n_features=450,
        alternate_sign=True,
        ngram_range=(1, 2)
    ),
    "HashingVectorizer_Ngram22": HashingVectorizer(
        n_features=600,
        alternate_sign=False,
        ngram_range=(2, 2)
    ),
}

CLASSIFIERS = {
    "LogisticRegression_SolverLbfgs": LogisticRegression(
        solver="lbfgs",
        C=0.5,
        max_iter=300
    ),
    "LogisticRegression_SolverSaga": LogisticRegression(
        solver="saga",
        C=0.3,
        max_iter=300
    ),
    "LogisticRegression_SolverLiblinear": LogisticRegression(
        solver="liblinear",
        C=0.7,
        max_iter=200
    ),
    "SVM_LinearKernel_C0.1": SVC(
        kernel="linear",
        C=0.1
    ),
    "SVM_LinearKernel_C0.5": SVC(
        kernel="linear",
        C=0.5
    ),
    "SVM_RbfKernel_C0.3": SVC(
        kernel="rbf",
        C=0.3
    ),
    "NaiveBayes_Alpha1.0": MultinomialNB(
        alpha=1.0
    ),
    "NaiveBayes_Alpha0.3": MultinomialNB(
        alpha=0.3
    ),
    "NaiveBayes_Alpha0.7": MultinomialNB(
        alpha=0.7
    ),
}

def process_model(vectorizer, vec_name, train_reviews, test_reviews, train_labels, test_labels, rootDir):
    train, test = transform_data(vectorizer, train_reviews, test_reviews)

    if train.shape[1] != test.shape[1]:
        print(f"Feature mismatch for {vec_name}: train ({train.shape[1]}), test ({test.shape[1]})")
        return []

    scaler = StandardScaler(with_mean=False)  # Skalowanie danych
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    vectorizer_filename = adjust_path(
        os.path.join(DICTIONARY.MODELS_PATH, DICTIONARY.VECTORIZER_PATH,
                     f"{vec_name}_{DICTIONARY.VECTORIZER_FILE}.{DICTIONARY.JOBLIB_EXTENSION}"),
        condition=rootDir,
    )
    save_vectorizer(vectorizer, vectorizer_filename)
    results = []

    for clf_name, classifier in CLASSIFIERS.items():
        try:
            model_filename = adjust_path(
                os.path.join(DICTIONARY.MODELS_PATH,
                             f"{clf_name}_{vec_name}_{DICTIONARY.MODEL_FILE}.{DICTIONARY.JOBLIB_EXTENSION}"),
                condition=rootDir,
            )
            classifier.fit(train, train_labels)
            try:
                save_model(classifier, model_filename)
            except Exception as e:
                print(f"Error saving model {clf_name} with vectorizer {vec_name}: {e}")
                continue

            train_pred = classifier.predict(train)
            test_pred = classifier.predict(test)

            results.extend(rate_model(train_pred, test_pred, vec_name, clf_name, train_labels, test_labels))
        except ValueError as e:
            print(f"Error training classifier {clf_name} with vectorizer {vec_name}: {e}")
            continue
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
