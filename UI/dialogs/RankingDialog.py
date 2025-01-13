from PyQt5.QtWidgets import QDialog, QVBoxLayout, QTableWidget, QTableWidgetItem, QMessageBox, QHeaderView


class ModelsRankingDialog(QDialog):
    def __init__(self, data):
        super().__init__()
        self.setWindowTitle("Models Ranking")
        self.setGeometry(150, 150, 800, 400)
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        self.table = QTableWidget()
        self.layout.addWidget(self.table)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.load_csv(data)

    def load_csv(self, data):
        self.table.setRowCount(len(data))
        self.table.setColumnCount(len(data.columns))
        self.table.setHorizontalHeaderLabels(data.columns)
        for row in range(len(data)):
            for col in range(len(data.columns)):
                self.table.setItem(row, col, QTableWidgetItem(str(data.iloc[row, col])))
