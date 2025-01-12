import sys
from PyQt5.QtWidgets import QApplication
from UI.UI import UIApp

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = UIApp()
    window.show()
    sys.exit(app.exec_())
