# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 
# PyQt5 UI code generator 5.10.1
#

from collections import OrderedDict
from genericpath import exists
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QDoubleValidator, QFont, QIcon, QFontMetrics
from PyQt5.QtCore import Qt
import json
from os import path, makedirs
from copy import deepcopy

import pyqt5.rc_ui  # pyrcc5 *.qrc -o *.py


class ConfigWindow(QtWidgets.QMainWindow):
    def __init__(self, mainWin, parent=None):
        super(ConfigWindow, self).__init__(parent)
        self.mainWin = mainWin
        self.model_dict = dict(json.load(open('./pyqt5/model_list.json', 'r')))
        self.custom_config = ''
        self.custom_path = path.join(path.abspath(path.curdir), 'results')
        self.setupUi()
    
    def setupUi(self):
        self.setObjectName("ConfigWindow")
        self.resize(840, 472.5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
        _screen_geo = QtWidgets.QDesktopWidget().screenGeometry()
        _self_geo = self.geometry()
        self.move(int((_screen_geo.width() - _self_geo.width()) / 2), \
            int((_screen_geo.height() - _self_geo.height()) / 2))

        self.centralwidget = QtWidgets.QWidget(self)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        font = QFont('Times New Roman')
        pointSize = font.pointSize()
        font.setPixelSize(pointSize * 80 / 36)
        fontHeight = QFontMetrics(font).size(0, '').height()

        self.horizontalLayout_general = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_general.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.horizontalLayout_general.setObjectName("horizontalLayout_general")

        self.verticalLayout_operate = QtWidgets.QVBoxLayout()
        self.verticalLayout_operate.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_operate.setSpacing(0)
        self.verticalLayout_operate.setObjectName("verticalLayout_operate")

        self.horizontalLayout_path = QtWidgets.QHBoxLayout()
        self.horizontalLayout_path.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_path.setSpacing(0)
        self.horizontalLayout_path.setObjectName("horizontalLayout_path")
        self.label_path = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_path.sizePolicy().hasHeightForWidth())
        self.label_path.setSizePolicy(sizePolicy)
        self.label_path.setObjectName("label_path")
        self.horizontalLayout_path.addWidget(self.label_path)
        self.textEdit_path = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.textEdit_path.sizePolicy().hasHeightForWidth())
        self.textEdit_path.setSizePolicy(sizePolicy)
        self.textEdit_path.setMaximumHeight(fontHeight + 8)
        self.textEdit_path.setObjectName("textEdit_path")
        self.horizontalLayout_path.addWidget(self.textEdit_path)
        self.pushButton_path = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_path.sizePolicy().hasHeightForWidth())
        self.pushButton_path.setSizePolicy(sizePolicy)
        self.pushButton_path.setMinimumWidth(5)
        self.pushButton_path.setMaximumHeight(fontHeight + 10)
        self.pushButton_path.setObjectName("pushButton_path")
        self.horizontalLayout_path.addWidget(self.pushButton_path)
        self.horizontalLayout_path.setStretch(0, 2)
        self.horizontalLayout_path.setStretch(1, 11)
        self.horizontalLayout_path.setStretch(2, 1)
        self.horizontalLayout_path.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_path)

        self.horizontalLayout_type = QtWidgets.QHBoxLayout()
        self.horizontalLayout_type.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_type.setSpacing(0)
        self.horizontalLayout_type.setObjectName("horizontalLayout_type")
        self.label_type = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_type.sizePolicy().hasHeightForWidth())
        self.label_type.setSizePolicy(sizePolicy)
        self.label_type.setObjectName("label_type")
        self.horizontalLayout_type.addWidget(self.label_type)
        _check_box_style = 'QCheckBox::indicator{width:20px;height:20px;}'
        self.checkBox_trace = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_trace.sizePolicy().hasHeightForWidth())
        self.checkBox_trace.setSizePolicy(sizePolicy)
        self.checkBox_trace.setStyleSheet(_check_box_style)
        self.checkBox_trace.setObjectName("checkBox_trace")
        self.horizontalLayout_type.addWidget(self.checkBox_trace)
        self.checkBox_bbox = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_bbox.sizePolicy().hasHeightForWidth())
        self.checkBox_bbox.setSizePolicy(sizePolicy)
        self.checkBox_bbox.setStyleSheet(_check_box_style)
        self.checkBox_bbox.setObjectName("checkBox_bbox")
        self.horizontalLayout_type.addWidget(self.checkBox_bbox)
        self.checkBox_mask = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_mask.sizePolicy().hasHeightForWidth())
        self.checkBox_mask.setSizePolicy(sizePolicy)
        self.checkBox_mask.setStyleSheet(_check_box_style)
        self.checkBox_mask.setObjectName("checkBox_mask")
        self.horizontalLayout_type.addWidget(self.checkBox_mask)
        self.checkBox_skeleton = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_skeleton.sizePolicy().hasHeightForWidth())
        self.checkBox_skeleton.setSizePolicy(sizePolicy)
        self.checkBox_skeleton.setStyleSheet(_check_box_style)
        self.checkBox_skeleton.setObjectName("checkBox_skeleton")
        self.horizontalLayout_type.addWidget(self.checkBox_skeleton)
        self.checkBox_raw = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_raw.sizePolicy().hasHeightForWidth())
        self.checkBox_raw.setSizePolicy(sizePolicy)
        self.checkBox_raw.setStyleSheet(_check_box_style)
        self.checkBox_raw.setObjectName("checkBox_raw")
        self.horizontalLayout_type.addWidget(self.checkBox_raw)
        self.checkBox_rawvideo = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_rawvideo.sizePolicy().hasHeightForWidth())
        self.checkBox_rawvideo.setSizePolicy(sizePolicy)
        self.checkBox_rawvideo.setStyleSheet(_check_box_style)
        self.checkBox_rawvideo.setObjectName("checkBox_rawvideo")
        self.horizontalLayout_type.addWidget(self.checkBox_rawvideo)
        self.horizontalLayout_type.setStretch(0, 1)
        self.horizontalLayout_type.setStretch(1, 1)
        self.horizontalLayout_type.setStretch(2, 1)
        self.horizontalLayout_type.setStretch(3, 1)
        self.horizontalLayout_type.setStretch(4, 1)
        self.horizontalLayout_type.setStretch(5, 1)
        self.horizontalLayout_type.setStretch(6, 1)
        self.horizontalLayout_type.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_type)

        self.horizontalLayout_set = QtWidgets.QHBoxLayout()
        self.horizontalLayout_set.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_set.setSpacing(0)
        self.horizontalLayout_set.setObjectName("horizontalLayout_set")
        self.label_clip = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_clip.sizePolicy().hasHeightForWidth())
        self.label_clip.setSizePolicy(sizePolicy)
        self.label_clip.setObjectName("label_clip")
        self.horizontalLayout_set.addWidget(self.label_clip)
        self.label_from = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_from.sizePolicy().hasHeightForWidth())
        self.label_from.setSizePolicy(sizePolicy)
        self.label_from.setObjectName("label_from")
        self.horizontalLayout_set.addWidget(self.label_from)
        self.spinBox_from = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.spinBox_from.sizePolicy().hasHeightForWidth())
        self.spinBox_from.setSizePolicy(sizePolicy)
        self.spinBox_from.setMaximumHeight(fontHeight + 8)
        self.spinBox_from.setObjectName("spinBox_from")
        self.horizontalLayout_set.addWidget(self.spinBox_from)
        self.label_to = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_to.sizePolicy().hasHeightForWidth())
        self.label_to.setSizePolicy(sizePolicy)
        self.label_to.setObjectName("label_to")
        self.horizontalLayout_set.addWidget(self.label_to)
        self.spinBox_to = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.spinBox_to.sizePolicy().hasHeightForWidth())
        self.spinBox_to.setSizePolicy(sizePolicy)
        self.spinBox_to.setMaximumHeight(fontHeight + 8)
        self.spinBox_to.setObjectName("spinBox_to")
        self.horizontalLayout_set.addWidget(self.spinBox_to)
        self.label_down = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_down.sizePolicy().hasHeightForWidth())
        self.label_down.setSizePolicy(sizePolicy)
        self.label_down.setObjectName("label_down")
        self.label_down.setAlignment(Qt.AlignCenter)
        self.horizontalLayout_set.addWidget(self.label_down)
        self.textEdit_down = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.textEdit_down.sizePolicy().hasHeightForWidth())
        self.textEdit_down.setSizePolicy(sizePolicy)
        self.textEdit_down.setMinimumWidth(10)
        self.textEdit_down.setMaximumHeight(fontHeight + 8)
        self.textEdit_down.setValidator(QDoubleValidator(0, 100, 2, self.centralwidget))
        self.textEdit_down.setObjectName("textEdit_down")
        self.horizontalLayout_set.addWidget(self.textEdit_down)
        self.horizontalLayout_set.setStretch(0, 4)
        self.horizontalLayout_set.setStretch(1, 2)
        self.horizontalLayout_set.setStretch(2, 5)
        self.horizontalLayout_set.setStretch(3, 1)
        self.horizontalLayout_set.setStretch(4, 5)
        self.horizontalLayout_set.setStretch(5, 8)
        self.horizontalLayout_set.setStretch(6, 3)
        self.horizontalLayout_set.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_set)

        self.horizontalLayout_vis = QtWidgets.QHBoxLayout()
        self.horizontalLayout_vis.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_vis.setSpacing(0)
        self.horizontalLayout_vis.setObjectName("horizontalLayout_vis")
        self.label_vis = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_vis.sizePolicy().hasHeightForWidth())
        self.label_vis.setSizePolicy(sizePolicy)
        self.label_vis.setObjectName("label_vis")
        self.horizontalLayout_vis.addWidget(self.label_vis)
        self.checkBox_vis = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_vis.sizePolicy().hasHeightForWidth())
        self.checkBox_vis.setSizePolicy(sizePolicy)
        self.checkBox_vis.setStyleSheet(_check_box_style)
        self.checkBox_vis.setObjectName("checkBox_vis")
        self.horizontalLayout_vis.addWidget(self.checkBox_vis)
        self.checkBox_fps = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_fps.sizePolicy().hasHeightForWidth())
        self.checkBox_fps.setSizePolicy(sizePolicy)
        self.checkBox_fps.setStyleSheet(_check_box_style)
        self.checkBox_fps.setObjectName("checkBox_fps")
        self.horizontalLayout_vis.addWidget(self.checkBox_fps)
        self.checkBox_ana = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_ana.sizePolicy().hasHeightForWidth())
        self.checkBox_ana.setSizePolicy(sizePolicy)
        self.checkBox_ana.setStyleSheet(_check_box_style)
        self.checkBox_ana.setObjectName("checkBox_ana")
        self.horizontalLayout_vis.addWidget(self.checkBox_ana)
        self.label_factor = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_factor.sizePolicy().hasHeightForWidth())
        self.label_factor.setSizePolicy(sizePolicy)
        self.label_factor.setObjectName("label_factor")
        self.label_factor.setAlignment(Qt.AlignCenter)
        self.horizontalLayout_vis.addWidget(self.label_factor)
        self.textEdit_factor = QtWidgets.QLineEdit(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.textEdit_factor.sizePolicy().hasHeightForWidth())
        self.textEdit_factor.setSizePolicy(sizePolicy)
        self.textEdit_factor.setMinimumWidth(10)
        self.textEdit_factor.setMaximumHeight(fontHeight + 8)
        self.textEdit_factor.setValidator(QDoubleValidator(0, 100, 2, self.centralwidget))
        self.textEdit_factor.setObjectName("textEdit_factor")
        self.horizontalLayout_vis.addWidget(self.textEdit_factor)
        self.horizontalLayout_vis.setStretch(0, 14)
        self.horizontalLayout_vis.setStretch(1, 10)
        self.horizontalLayout_vis.setStretch(2, 7)
        self.horizontalLayout_vis.setStretch(3, 10)
        self.horizontalLayout_vis.setStretch(4, 9)
        self.horizontalLayout_vis.setStretch(5, 6)
        self.horizontalLayout_vis.setContentsMargins(10, 10, 10, 10)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_vis)

        self.verticalLayout_custom = QtWidgets.QVBoxLayout()
        self.verticalLayout_custom.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_custom.setSpacing(0)
        self.verticalLayout_custom.setObjectName("verticalLayout_custom")
        self.horizontalLayout_custom = QtWidgets.QHBoxLayout()
        self.horizontalLayout_custom.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_custom.setSpacing(0)
        self.horizontalLayout_custom.setObjectName("horizontalLayout_custom")
        self.label_custom = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_custom.sizePolicy().hasHeightForWidth())
        self.label_custom.setSizePolicy(sizePolicy)
        self.label_custom.setObjectName("label_custom")
        self.label_custom.setAlignment(Qt.AlignCenter)
        self.horizontalLayout_custom.addWidget(self.label_custom)
        self.checkBox_custom = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox_custom.sizePolicy().hasHeightForWidth())
        self.checkBox_custom.setSizePolicy(sizePolicy)
        self.checkBox_custom.setStyleSheet(_check_box_style)
        self.checkBox_custom.setObjectName("checkBox_custom")
        self.horizontalLayout_custom.addWidget(self.checkBox_custom)
        self.horizontalLayout_custom.setStretch(0, 3)
        self.horizontalLayout_custom.setStretch(1, 1)
        self.horizontalLayout_custom.setContentsMargins(10, 0, 10, 0)
        self.verticalLayout_custom.addLayout(self.horizontalLayout_custom)
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName("textBrowser")
        self.textBrowser.setReadOnly(False)
        self.textBrowser.setUndoRedoEnabled(True)
        self.verticalLayout_custom.addWidget(self.textBrowser)
        self.verticalLayout_custom.setStretch(0, 1)
        self.verticalLayout_custom.setStretch(1, 9)
        self.verticalLayout_custom.setContentsMargins(5, 10, 5, 10)
        self.verticalLayout_operate.addLayout(self.verticalLayout_custom)

        self.verticalLayout_operate.setStretch(0, 1)
        self.verticalLayout_operate.setStretch(1, 1)
        self.verticalLayout_operate.setStretch(2, 1)
        self.verticalLayout_operate.setStretch(3, 1)
        self.verticalLayout_operate.setStretch(4, 10)
        self.horizontalLayout_general.addLayout(self.verticalLayout_operate)
        
        self.setCentralWidget(self.centralwidget)
        self.label_custom.setFont(font)
        self.label_clip.setFont(font)
        self.label_down.setFont(font)
        self.label_from.setFont(font)
        self.label_to.setFont(font)
        self.label_path.setFont(font)
        self.label_type.setFont(font)
        self.label_vis.setFont(font)
        self.label_factor.setFont(font)
        self.pushButton_path.setFont(font)
        self.textEdit_down.setFont(font)
        self.textEdit_path.setFont(font)
        self.textEdit_factor.setFont(font)
        self.textBrowser.setFont(font)
        self.checkBox_trace.setFont(font)
        self.checkBox_bbox.setFont(font)
        self.checkBox_mask.setFont(font)
        self.checkBox_skeleton.setFont(font)
        self.checkBox_raw.setFont(font)
        self.checkBox_rawvideo.setFont(font)
        self.checkBox_custom.setFont(font)
        self.checkBox_vis.setFont(font)
        self.checkBox_fps.setFont(font)
        self.checkBox_ana.setFont(font)
        self.spinBox_from.setFont(font)
        self.spinBox_to.setFont(font)
        self.setFont(font)
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, ConfigWindow):
        _translate = QtCore.QCoreApplication.translate
        ConfigWindow.setWindowTitle(_translate("ConfigWindow", "Output Configuring"))
        ConfigWindow.setWindowIcon(QIcon(':rc/icon.png'))  # Thanks Open Source Site: https://www.iconfont.cn/
        self.label_path.setText(' Path: ')
        self.label_type.setText(' Type: ')
        self.label_clip.setText(' Clip: ')
        self.label_from.setText('from ')
        self.label_to.setText('  to ')
        self.label_down.setText('    Downsample: ')
        self.label_custom.setText('Custom Initialization')
        self.label_vis.setText(' Visualization: ')
        self.label_factor.setText('  Factor:  ')
        self.textEdit_down.setText('1.00')
        self.pushButton_path.setText(_translate("ConfigWindow", "⋯"))
        self.checkBox_trace.setText(_translate("ConfigWindow", "&trace"))
        self.checkBox_bbox.setText(_translate("ConfigWindow", "&b-box"))
        self.checkBox_mask.setText(_translate("ConfigWindow", "&mask"))
        self.checkBox_skeleton.setText(_translate("ConfigWindow", "&skel"))
        self.checkBox_raw.setText(_translate("ConfigWindow", "raw"))
        self.checkBox_rawvideo.setText(_translate("ConfigWindow", "raw-v"))
        self.checkBox_vis.setText(_translate("ConfigWindow", "activate"))
        self.checkBox_vis.setChecked(True)
        self.checkBox_fps.setText(_translate("ConfigWindow", "fps"))
        self.checkBox_ana.setText(_translate("ConfigWindow", "analysis"))
        self.checkBox_custom.setText(_translate("ConfigWindow", "use the following"))
    

    def showEvent(self, event):
        # Load defalut output path
        self.textEdit_path.setText(self.custom_path)
        # Re-enable
        self.checkBox_trace.setEnabled(True)
        self.checkBox_bbox.setEnabled(True)
        self.checkBox_mask.setEnabled(True)
        self.checkBox_skeleton.setEnabled(True)
        # Ban model output type
        _output_list = self.model_dict[self.mainWin.model_name]
        if 'trace' not in _output_list:
            self.checkBox_trace.setEnabled(False)
        if 'bbox' not in _output_list:
            self.checkBox_bbox.setEnabled(False)
        if 'mask' not in _output_list:
            self.checkBox_mask.setEnabled(False)
        if 'skeleton' not in _output_list:
            self.checkBox_skeleton.setEnabled(False)
        # Print b-box setting
        if not self.checkBox_custom.isChecked():
            self.textBrowser.clear()
            _bbox_setting_text = ''
            if self.mainWin.INIT in self.mainWin.info_store.keys():
                _bbox_setting_text = json.dumps(self.mainWin.info_store[self.mainWin.INIT])
                _new_text = ''
                for c in _bbox_setting_text:
                    c_d = deepcopy(c)
                    if c == '{':
                        c_d += '\n\t '
                    elif c == '}':
                        c_d = '\n' + c_d
                    elif c == ':':
                        c_d += '\t'
                    elif c == ',' and _new_text[-1] == ']':
                        c_d += '\n\t'
                    _new_text += c_d
                _bbox_setting_text = _new_text
            self.textBrowser.append(_bbox_setting_text)
        self.textBrowser.textChanged.connect(self.custom_change)
        # Initialize clip range
        if self.mainWin.video_name == '':
            self.spinBox_from.setValue(0)
            self.spinBox_to.setValue(0)
            self.spinBox_from.setEnabled(False)
            self.spinBox_to.setEnabled(False)
        else:
            self.spinBox_from.setEnabled(True)
            self.spinBox_to.setEnabled(True)
            self.spinBox_from.setMaximum(self.mainWin.F - 1)
            self.spinBox_to.setMaximum(self.mainWin.F - 1)
            self.spinBox_to.setValue(self.mainWin.F - 1)
        # QLabel operations
        self.checkBox_vis.stateChanged.connect(self.visible_change)
        self.textEdit_factor.setText('{:.2f}'.format(self.mainWin.video_factor))

    def closeEvent(self, event):
        if not path.exists(self.custom_path):
            makedirs(self.custom_path)
        # Return custom setting
        if self.checkBox_custom.isChecked():
            try:
                if self.mainWin.first_frame:
                    print('[INFO] Setting from custom:', self.custom_config)
                    _brow_setting_dict = dict(json.loads(self.custom_config))
                    _new_dict = {}
                    for id, bbox in _brow_setting_dict.items():
                        _new_dict[int(id)] = bbox
                    self.mainWin.info_store[self.mainWin.INIT] = _new_dict
            except:
                print('[INFO] Error for setting above custom.')
                QtWidgets.QMessageBox.information(self, 'Warning', 'Error When Use Custom Settings ' +
                                                  '(See JSON File Format).', QtWidgets.QMessageBox.Ok)
                event.ignore()
                return
        self.textBrowser.textChanged.disconnect(self.custom_change)
        # QLabel operations
        self.mainWin.video_activate = self.checkBox_vis.isChecked()
        self.checkBox_vis.stateChanged.disconnect(self.visible_change)
        self.mainWin.video_factor = min(float(self.textEdit_factor.text()), 100.0)
        self.mainWin.video_fps = self.checkBox_fps.isChecked()
        # Re-enable
        self.setEnabled(False)
        self.mainWin.setEnabled(True)
    
    def custom_change(self):
        _brow_setting_text = self.textBrowser.toPlainText()
        _new_text = ''
        for c in _brow_setting_text:
            if c == '\n':
                pass
            elif c == '\t':
                pass
            elif c == ' ' and _new_text[-1] == '{':
                pass
            else:
                _new_text += c
        self.custom_config = _new_text

    def visible_change(self):
        if self.checkBox_vis.isChecked():
            self.checkBox_raw.setEnabled(True)
            self.checkBox_rawvideo.setEnabled(True)
            self.checkBox_fps.setEnabled(True)
            self.checkBox_ana.setEnabled(True)
        else:
            self.checkBox_raw.setChecked(False)
            self.checkBox_rawvideo.setChecked(False)
            self.checkBox_fps.setChecked(False)
            self.checkBox_ana.setChecked(False)
            self.checkBox_raw.setEnabled(False)
            self.checkBox_rawvideo.setEnabled(False)
            self.checkBox_fps.setEnabled(False)
            self.checkBox_ana.setEnabled(False)
            print('[INFO] Deactivate the GUI image showing.')
            QtWidgets.QMessageBox.information(self, 'Warning', 'Deactivate Visualization to ' +
                                              'Accelerate Algorithm Process.', QtWidgets.QMessageBox.Ok)


'''
a_sample_of_custom_initialization:
{"1": [425, 508, 164, 80], 
"2": [935, 549, 83, 151], 
"3": [905, 389, 67, 135], 
"4": [872, 724, 156, 65], 
"5": [253, 688, 132, 77], 
"6": [170, 491, 129, 127]}
'''
