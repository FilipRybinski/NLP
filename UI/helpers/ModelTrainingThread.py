from PyQt5.QtCore import QThread, pyqtSignal

from scripts.utils import create_models


class ModelTrainingThread(QThread):
    training_complete = pyqtSignal()

    def run(self):
        create_models(False)
        self.training_complete.emit()