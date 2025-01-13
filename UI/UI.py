from PyQt5.QtWidgets import QMainWindow, QAction, QTextEdit, QLabel, QPushButton, QWidget, QVBoxLayout, QMenuBar
from UI.dialogs.utils import load_ranking_data, get_best_model_and_vectorizer
from UI.utils.actions import open_models_dialog, open_models_ranking_table, process_preview, clear_inputs
from constants.constants import DICTIONARY


class UIApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Movie Review App")

        # Properties
        self.model = None
        self.vectorizer = None
        self.classifier_name = None
        self.vectorizer_name = None
        self.data_ranking = load_ranking_data(self,f"{DICTIONARY.MODELS_RANKING_PATH}/{DICTIONARY.CLASSIFICATION_RESULT_FILE}")
        get_best_model_and_vectorizer(self)

        # Main layout
        self.set_main_layout_properties()

        # Menu bar
        self.create_navbar()

        # Review label and text area
        self.create_input_review()

        # Buttons
        self.create_action_buttons()

    def create_navbar(self):
        # Initialize the menu bar
        self.menu_bar = QMenuBar()
        self.setMenuBar(self.menu_bar)

        # File menu
        self.file_menu = self.menu_bar.addMenu("Menu")

        # Add "Load models" action
        self.open_file_dialog_action = QAction("Load models", self)
        self.open_file_dialog_action.triggered.connect(lambda: open_models_dialog(self))
        self.file_menu.addAction(self.open_file_dialog_action)

        # Add "Show models ranking" action
        self.open_csv_table_action = QAction("Show models ranking", self)
        self.open_csv_table_action.triggered.connect(lambda: open_models_ranking_table(self))
        self.file_menu.addAction(self.open_csv_table_action)

    def create_input_review(self):
        self.model_info = QLabel(f"Classifier : {self.classifier_name}")
        self.layout.addWidget(self.model_info)

        self.vectorizer_info = QLabel(f"Current vectorizer: {self.vectorizer_name}")
        self.layout.addWidget(self.vectorizer_info)

        # Label and text area for review input
        self.review_label = QLabel("Your Review:")
        self.layout.addWidget(self.review_label)

        self.review_input = QTextEdit()
        self.review_input.setPlaceholderText("Write your review here...")
        self.layout.addWidget(self.review_input)

    def create_action_buttons(self):
        # Process Review button
        self.save_button = QPushButton("Process Review")
        self.save_button.clicked.connect(lambda: process_preview(self))
        self.layout.addWidget(self.save_button)

        # Clear button
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(lambda: clear_inputs(self))
        self.layout.addWidget(self.clear_button)

    def set_main_layout_properties(self):
        # Set up the main layout
        self.main_widget = QWidget()
        self.setCentralWidget(self.main_widget)

        self.layout = QVBoxLayout()
        self.main_widget.setLayout(self.layout)

