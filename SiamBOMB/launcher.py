# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 

import sys
import torch
from os import path

from PyQt5.QtWidgets import QApplication
from .pyqt5.window_main import SiamBOMBWindow

torch.set_num_threads(24)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWin = SiamBOMBWindow()

    if mainWin.isDracula:
        # Open source from https://github.com/Kojoley/qss-dracula
        with open(path.join(path.dirname(path.abspath(__file__)), \
            'pyqt5/dracula.qss'), 'r') as f:
            qssStyle = f.read()
        mainWin.setStyleSheet(qssStyle)

    mainWin.show()
    sys.exit(app.exec_())
