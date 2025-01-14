from tkinter import Place

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QMainWindow, QAction, QTextEdit, QLabel, QPushButton, QWidget, QVBoxLayout, QMenuBar, QMessageBox, QProgressDialog
)
from UI.dialogs.ModelsDialog import ModelsDialog
from UI.dialogs.RankingDialog import ModelsRankingDialog
from UI.helpers.ModelTrainingThread import ModelTrainingThread
from UI.utils import predict_review, load_model, load_csv_data, get_best_csv_classifier_vectorizer
from constants.errors import Errors
from constants.placeholders import Placeholders


class UIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(Placeholders.APP_NAME)

        self.classifier = None
        self.vectorizer = None
        self.classifier_name = None
        self.vectorizer_name = None

        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        self.file_menu = self.menu_bar.addMenu(Placeholders.MENU)

        self.open_file_dialog_action = QAction(Placeholders.LOAD_MODELS, self)
        self.open_file_dialog_action.triggered.connect(self.open_models_dialog)
        self.file_menu.addAction(self.open_file_dialog_action)

        self.open_csv_table_action = QAction(Placeholders.SHOW_RANKING, self)
        self.open_csv_table_action.triggered.connect(self.open_models_ranking_table)
        self.file_menu.addAction(self.open_csv_table_action)

        self.train_models_action = QAction(Placeholders.TRAIN, self)
        self.train_models_action.triggered.connect(self.train_models)
        self.file_menu.addAction(self.train_models_action)

        self.classifier_info = QLabel(Placeholders.CLASSIFIER_LABEL)
        self.layout.addWidget(self.classifier_info)

        self.vectorizer_info = QLabel(Placeholders.VECTORIZER_LABEL)
        self.layout.addWidget(self.vectorizer_info)

        self.review_label = QLabel(Placeholders.REVIEW_LABEL)
        self.layout.addWidget(self.review_label)

        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText(Placeholders.WRITE_STH)
        self.layout.addWidget(self.review_input)

        self.save_button = QPushButton(Placeholders.PROCESS_REVIEW)
        self.save_button.clicked.connect(self.process_review)
        self.layout.addWidget(self.save_button)

        self.clear_button = QPushButton(Placeholders.CLEAR)
        self.clear_button.clicked.connect(self.clear_inputs)
        self.layout.addWidget(self.clear_button)

        self.classifier_name, self.vectorizer_name, self.classifier, self.vectorizer = get_best_csv_classifier_vectorizer()
        self.set_info()

    def process_review(self):
        review = self.review_input.toPlainText().strip()
        if not review:
            QMessageBox.critical(self,  Errors.Error, Errors.REVIEW_EMPTY )
            return

        if not self.classifier or not self.vectorizer:
            QMessageBox.critical(self,  Errors.Error, Errors.MODEL_NOT_LOADED)
            return

        result = predict_review(self.classifier, self.vectorizer, review)
        QMessageBox.information(self, Placeholders.REVIEW_INFO, f"Review is {result.lower()}")

    def open_models_dialog(self):
        file_dialog = ModelsDialog()
        if file_dialog.exec_():
            if file_dialog.classifier_name and file_dialog.vectorizer_name:
                self.classifier, self.vectorizer = load_model(file_dialog.classifier_path, file_dialog.vectorizer_path)
                self.classifier_name = file_dialog.classifier_name
                self.vectorizer_name = file_dialog.vectorizer_name
                self.set_info()
                QMessageBox.information(self, Placeholders.MODEL_LOADED, Errors.LOADED_SUCCESSFULLY)

    def open_models_ranking_table(self):
        data = load_csv_data()
        try:
            if data is None:
                raise ValueError(Errors.NO_RANKING)

            dialog = ModelsRankingDialog(data)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, Errors.Error, f"Failed to open models ranking table: {e}")

    def clear_inputs(self):
        self.review_input.clear()

    def train_models(self):
        self.progress_dialog = QProgressDialog(Placeholders.TRAINING_PROG_CDN, None, 0, 0, self)
        self.progress_dialog.setWindowTitle(Placeholders.TRAINING_PROG)
        self.progress_dialog.setWindowModality(Qt.ApplicationModal)
        self.progress_dialog.setCancelButton(None)
        self.progress_dialog.setRange(0, 0)
        self.progress_dialog.show()

        self.training_thread = ModelTrainingThread()
        self.training_thread.training_complete.connect(self.finish_training)
        self.training_thread.start()

    def finish_training(self):
        self.progress_dialog.close()
        QMessageBox.information(self, Placeholders.TRAINING_COMPLETED, Errors.MODELS_SUCCESSFULLY_TRAINED)
        self.training_thread = None

    def set_info(self):
        self.classifier_info.setText(f"Classifier: {self.classifier_name}")
        self.vectorizer_info.setText(f"Current vectorizer: {self.vectorizer_name}")
