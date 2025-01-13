from PyQt5.QtWidgets import QMessageBox
from UI.dialogs.ModelsDialog import ModelsDialog
from UI.dialogs.RankingDialog import ModelsRankingDialog
from utils.utils import predict_review, load_model


def process_preview(self):
    review = self.review_input.toPlainText().strip()
    if not review:
        QMessageBox.critical(self, "Error", "Review cannot be empty!")
        return
    if not self.model or not self.vectorizer:
        print(self.model, self.vectorizer)
        return
    result = predict_review(self, review)
    QMessageBox.information(self, "Review information", f"Review is {result.lower()}")

def open_models_dialog(self):
        self.file_dialog = ModelsDialog()
        if self.file_dialog.exec_():
            if self.file_dialog.classifier_name and self.file_dialog.vectorizer_name:
                self.model_info.setText(f"Classifier: {self.file_dialog.classifier_name}")
                self.vectorizer_info.setText(f"Current vectorizer: {self.file_dialog.vectorizer_name}")
                load_model(self, self.file_dialog.classifier_path, self.file_dialog.vectorizer_path)
                QMessageBox.information(self, "Model Loaded", f"Loaded model")

def open_models_ranking_table(self):
        try:
            dialog = ModelsRankingDialog(self.data_ranking)
            dialog.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open CSV Table: {e}")

def clear_inputs(self):
        self.review_input.clear()