## UTILS
import os
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump, load
from constants.constants import  VECTORIZERS, CLASSIFIERS, DICTIONARY

def create_models():
    train_reviews, train_labels = load_data(os.path.join(DICTIONARY.DATA_PATH, DICTIONARY.TRAIN_PATH))
    test_reviews, test_labels = load_data(os.path.join(DICTIONARY.DATA_PATH, DICTIONARY.TEST_PATH))
    make_model_dir()
    for vec_name, vectorizer in VECTORIZERS.items():
        train,test = transform_data(vectorizer,train_reviews,test_reviews)
        save_vectorizer(vectorizer, f"{DICTIONARY.MODELS_PATH}/{DICTIONARY.VECTORIZER_PATH}/{vec_name}_{DICTIONARY.VECTORIZER_FILE}.{DICTIONARY.JOBLIB_EXTENSION}")
        for clf_name, classifier in CLASSIFIERS.items():
            model = save_model(classifier,f"{DICTIONARY.MODELS_PATH}/{clf_name}_{vec_name}_{DICTIONARY.MODEL_FILE}.{DICTIONARY.JOBLIB_EXTENSION}")
            classifier.fit(train, train_labels)
            rate_model(classifier.predict(train), model.predict(test), vec_name, clf_name,train_labels, test_labels)


def load_data(data_dir):
    reviews, labels = [], []
    for label in ["pos", "neg"]:
        folder = os.path.join(data_dir, label)
        for filename in os.listdir(folder):
            with open(os.path.join(folder, filename), encoding="utf-8") as f:
                reviews.append(f.read())
                labels.append(1 if label == "pos" else 0)
    return reviews, labels

def make_model_dir():
    os.makedirs(f"../{DICTIONARY.MODELS_PATH}", exist_ok=True)

def transform_data(vectorizer,train_reviews,test_reviews):
    return vectorizer.fit_transform(train_reviews), vectorizer.transform(test_reviews)

def save_vectorizer(vectorizer,vectorizer_filename):
    dump(vectorizer, vectorizer_filename)

def save_model(classifier,model_filename):
    dump(classifier, model_filename)
    return load(model_filename)

def rate_model(train_pred, test_pred, vec_name, clf_name,train_labels, test_labels):
    results = []
    train_accuracy = accuracy_score(train_labels, train_pred)
    test_accuracy = accuracy_score(test_labels, test_pred)
    test_f1 = f1_score(test_labels, test_pred)
    results.append({
        DICTIONARY.VECTORIZER: vec_name,
        DICTIONARY.CLASSIFIER: clf_name,
        DICTIONARY.TRAIN_ACCURACY: train_accuracy,
        DICTIONARY.TEST_ACCURACY: test_accuracy,
        DICTIONARY.TEST_F1_SCORE: test_f1,
    })
    display_and_save_rate_model(results)


def display_and_save_rate_model(results):
    results_df = pd.DataFrame(results)
    print(results_df.sort_values(by="Test Accuracy", ascending=False))
    results_df.to_csv(DICTIONARY.CLASSIFICATION_RESULT_FILE, index=False)

def predict_review(self, review):
    try:
        review_vector = self.vectorizer.transform([review])
        prediction = self.model.predict(review_vector)
        return DICTIONARY.POSITIVE if prediction[0] == 1 else DICTIONARY.NEGATIVE
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None

def load_model(self,model_path, vectorizer_path):
    self.model = load(model_path)
    self.vectorizer = load(vectorizer_path)
