import os

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QListWidget, QLabel, QHBoxLayout, QPushButton, QMessageBox

from constants.constants import DICTIONARY


class ModelsDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.classifier_name = None
        self.classifier_path = None
        self.vectorizer_name = None
        self.vectorizer_path = None

        self.setWindowTitle("Models")
        self.setGeometry(150, 150, 400, 300)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.file_list_label = QLabel("Available Models")
        self.layout.addWidget(self.file_list_label)

        self.file_list = QListWidget()
        self.layout.addWidget(self.file_list)

        self.load_files()
        button_layout = QHBoxLayout()
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_selection)
        button_layout.addWidget(self.apply_button)

        self.layout.addLayout(button_layout)
        self.file_list.itemDoubleClicked.connect(self.select_file)

    def load_files(self):
        directory = os.path.join(os.getcwd(), DICTIONARY.MODELS_PATH)
        if os.path.exists(directory):
            self.file_list.clear()
            try:
                files = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
                self.file_list.addItems(files)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to list files: {e}")
        else:
            QMessageBox.critical(self, "Error", "The 'models' directory does not exist!")

    def select_file(self, item):
        self.set_properties(item.text())
        QMessageBox.information(self, "File Selected", f"You selected: {self.classifier_path}")
        self.accept()

    def apply_selection(self):
        selected_item = self.file_list.currentItem()
        if selected_item:
            self.set_properties(selected_item.text())
            QMessageBox.information(self, "File Selected", f"You selected: {self.classifier_path}")
            self.accept()
        else:
            QMessageBox.warning(self, "No Selection", "Please select a file from the list.")

    def set_properties(self, name):
        directory = os.path.join(os.getcwd(), DICTIONARY.MODELS_PATH)
        vectorizer_name = name.split("_")[1]
        classifier_name = name.split("_")[0]
        self.classifier_name = classifier_name
        self.vectorizer_name = vectorizer_name
        self.classifier_path = os.path.join(directory, name)
        self.vectorizer_path = os.path.join(DICTIONARY.MODELS_PATH,DICTIONARY.VECTORIZER_PATH,f"{vectorizer_name}_{DICTIONARY.VECTORIZER_FILE}.{DICTIONARY.JOBLIB_EXTENSION}")