# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 
# PyQt5 UI code generator 5.10.1
#

from collections import OrderedDict
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtGui import QPainter, QPixmap, QPen, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint
from copy import deepcopy
import cv2
import json

import pyqt5.rc_ui  # pyrcc5 *.qrc -o *.py


class Rect:
    def __init__(self):
        self.start = QPoint()
        self.end = QPoint()

    def setStart(self, s):
        self.start = s

    def setEnd(self, e):
        self.end = e

    def startPoint(self):
        return self.start

    def endPoint(self):
        return self.end

    def paint(self, painter):
        painter.drawRect(self.startPoint().x(), self.startPoint().y(),
                         self.endPoint().x() - self.startPoint().x(),
                         self.endPoint().y() - self.startPoint().y())


class Ui_MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # Some basal settings of the stream
        self.W = 800
        self.H = 600
        self.F = 0
        self.INIT = 0
        self.N = 0
        # Some initialization of the painter
        self.setMouseTracking(True)
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        self.info = {}
        self.info_store = OrderedDict()
        self.pp = QPainter()  # assistant painter
        self.paint_frame = None  # cv2 painting
        self.tempPix = QPixmap(self.W, self.H)  # qt painting
        self.tempPix.fill(Qt.white)
        self.shape = None
        self.rectList = []
        self.perm = False
        self.first_frame = True  # Is now first time to run this data
        self.isPainting = False  # Is now in painting mode
        self.ctrlPressed = False
        self.setupUi()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1920, 1080)
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
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetMinAndMaxSize)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.horizontalLayout_general = QtWidgets.QHBoxLayout()
        self.horizontalLayout_general.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_general.setObjectName("horizontalLayout_general")
        self.verticalLayout_stream = QtWidgets.QVBoxLayout()
        self.verticalLayout_stream.setSpacing(0)
        self.verticalLayout_stream.setObjectName("verticalLayout_stream")
        self.label_image = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_image.sizePolicy().hasHeightForWidth())
        self.label_image.setSizePolicy(sizePolicy)
        self.label_image.setMinimumSize(QtCore.QSize(400, 400))
        self.label_image.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image.setObjectName("label_image")
        self.verticalLayout_stream.addWidget(self.label_image)
        self.scrollBar = QtWidgets.QScrollBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.scrollBar.sizePolicy().hasHeightForWidth())
        self.scrollBar.setSizePolicy(sizePolicy)
        self.scrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.scrollBar.setObjectName("scrollBar")
        self.scrollBar.setMaximum(0)
        self.verticalLayout_stream.addWidget(self.scrollBar)
        self.verticalLayout_stream.setStretch(0, 24)
        self.verticalLayout_stream.setStretch(1, 1)
        self.horizontalLayout_general.addLayout(self.verticalLayout_stream)
        self.verticalLayout_operate = QtWidgets.QVBoxLayout()
        self.verticalLayout_operate.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_operate.setSpacing(0)
        self.verticalLayout_operate.setObjectName("verticalLayout_operate")
        self.horizontalLayout_logo = QtWidgets.QHBoxLayout()
        self.horizontalLayout_logo.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_logo.setContentsMargins(-1, -1, -1, 0)
        self.horizontalLayout_logo.setSpacing(0)
        self.horizontalLayout_logo.setObjectName("horizontalLayout_logo")
        self.label_logo = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_logo.sizePolicy().hasHeightForWidth())
        self.label_logo.setSizePolicy(sizePolicy)
        self.label_logo.setAlignment(Qt.AlignCenter)
        self.label_logo.setObjectName("label_logo")
        self.horizontalLayout_logo.addWidget(self.label_logo)
        self.horizontalLayout_logo.setStretch(0, 20)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_logo)

        self.horizontalLayout_model = QtWidgets.QHBoxLayout()
        self.horizontalLayout_model.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_model.setContentsMargins(-1, 15, -1, 20)
        self.horizontalLayout_model.setObjectName("horizontalLayout_model")
        self.label_comboBox = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_comboBox.sizePolicy().hasHeightForWidth())
        self.label_comboBox.setSizePolicy(sizePolicy)
        self.label_comboBox.setObjectName("label_comboBox")
        self.horizontalLayout_model.addWidget(self.label_comboBox)
        self.comboBox = QtWidgets.QComboBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.comboBox.sizePolicy().hasHeightForWidth())
        self.comboBox.setSizePolicy(sizePolicy)
        self.comboBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.comboBox.setObjectName('comboBox')
        self.comboBox.setEditable(True)
        self.horizontalLayout_model.addWidget(self.comboBox)
        self.horizontalLayout_model.setStretch(0, 3)
        self.horizontalLayout_model.setStretch(1, 7)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_model)

        self.verticalLayout_button = QtWidgets.QVBoxLayout()
        self.verticalLayout_button.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_button.setSpacing(15)
        self.verticalLayout_button.setObjectName("verticalLayout_button")
        self.horizontalLayout_loading = QtWidgets.QHBoxLayout()
        self.horizontalLayout_loading.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_loading.setObjectName("horizontalLayout_loading")
        self.pushButton_locationLoading = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_locationLoading.sizePolicy().hasHeightForWidth())
        self.pushButton_locationLoading.setSizePolicy(sizePolicy)
        self.pushButton_locationLoading.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_locationLoading.setObjectName("pushButton_locationLoading")
        self.horizontalLayout_loading.addWidget(self.pushButton_locationLoading)
        self.pushButton_videoLoading = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_videoLoading.sizePolicy().hasHeightForWidth())
        self.pushButton_videoLoading.setSizePolicy(sizePolicy)
        self.pushButton_videoLoading.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_videoLoading.setObjectName("pushButton_videoLoading")
        self.horizontalLayout_loading.addWidget(self.pushButton_videoLoading)
        self.verticalLayout_button.addLayout(self.horizontalLayout_loading)
        self.horizontalLayout_loading.setStretch(0, 7)
        self.horizontalLayout_loading.setStretch(1, 6)
        self.pushButton_outputConfiguring = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_outputConfiguring.sizePolicy().hasHeightForWidth())
        self.pushButton_outputConfiguring.setSizePolicy(sizePolicy)
        self.pushButton_outputConfiguring.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_outputConfiguring.setObjectName("pushButton_outputConfiguring")
        self.verticalLayout_button.addWidget(self.pushButton_outputConfiguring)
        self.pushButton_bboxSetting = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_bboxSetting.sizePolicy().hasHeightForWidth())
        self.pushButton_bboxSetting.setSizePolicy(sizePolicy)
        self.pushButton_bboxSetting.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_bboxSetting.setObjectName("pushButton_bboxSetting")
        self.verticalLayout_button.addWidget(self.pushButton_bboxSetting)
        self.pushButton_algorithmProcessing = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_algorithmProcessing.sizePolicy().hasHeightForWidth())
        self.pushButton_algorithmProcessing.setSizePolicy(sizePolicy)
        self.pushButton_algorithmProcessing.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_algorithmProcessing.setObjectName("pushButton_algorithmProcessing")
        self.verticalLayout_button.addWidget(self.pushButton_algorithmProcessing)
        self.verticalLayout_operate.addLayout(self.verticalLayout_button)

        self.horizontalLayout_check = QtWidgets.QHBoxLayout()
        self.horizontalLayout_check.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_check.setSpacing(0)
        self.horizontalLayout_check.setObjectName("horizontalLayout_check")
        self.label_spinBox = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_spinBox.sizePolicy().hasHeightForWidth())
        self.label_spinBox.setSizePolicy(sizePolicy)
        self.label_spinBox.setObjectName("label_spinBox")
        self.horizontalLayout_check.addWidget(self.label_spinBox)
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.spinBox.sizePolicy().hasHeightForWidth())
        self.spinBox.setSizePolicy(sizePolicy)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_check.addWidget(self.spinBox)
        self.label_selectBox = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_selectBox.sizePolicy().hasHeightForWidth())
        self.label_selectBox.setSizePolicy(sizePolicy)
        self.label_selectBox.setObjectName("label_selectBox")
        self.horizontalLayout_check.addWidget(self.label_selectBox)
        self.selectBox = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.selectBox.sizePolicy().hasHeightForWidth())
        self.selectBox.setSizePolicy(sizePolicy)
        self.selectBox.setMaximum(16777215)
        self.selectBox.setObjectName("selectBox")
        self.horizontalLayout_check.addWidget(self.selectBox)
        self.horizontalLayout_check.setStretch(0, 1)
        self.horizontalLayout_check.setStretch(1, 2)
        self.horizontalLayout_check.setStretch(2, 2)
        self.horizontalLayout_check.setStretch(3, 3)
        self.horizontalLayout_check.setContentsMargins(-1, 15, -1, 20)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_check)

        self.verticalLayout_message = QtWidgets.QVBoxLayout()
        self.verticalLayout_message.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.verticalLayout_message.setSpacing(10)
        self.verticalLayout_message.setObjectName("verticalLayout_message")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_message.addWidget(self.textBrowser)
        self.progressBar = QtWidgets.QProgressBar(self.centralwidget)
        fontP = QFont('Times New Roman')
        pointSizeP = fontP.pointSize()
        fontP.setPixelSize(pointSizeP * 80 / 36)
        self.progressBar.setFont(fontP)
        self.progressBar.setMouseTracking(False)
        self.progressBar.setProperty("value", 0)
        self.progressBar.setAlignment(QtCore.Qt.AlignCenter)
        self.progressBar.setInvertedAppearance(False)
        self.progressBar.setTextDirection(QtWidgets.QProgressBar.TopToBottom)
        self.progressBar.setObjectName("progressBar")
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.progressBar.setSizePolicy(sizePolicy)
        self.verticalLayout_message.addWidget(self.progressBar)
        self.verticalLayout_message.setStretch(0, 10)
        self.verticalLayout_message.setStretch(1, 1)
        self.verticalLayout_operate.addLayout(self.verticalLayout_message)

        self.verticalLayout_operate.setStretch(0, 10)
        self.verticalLayout_operate.setStretch(1, 2)
        self.verticalLayout_operate.setStretch(2, 16)
        self.verticalLayout_operate.setStretch(3, 2)
        self.verticalLayout_operate.setStretch(4, 25)
        self.horizontalLayout_general.addLayout(self.verticalLayout_operate)
        self.horizontalLayout_general.setStretch(0, 3)
        self.horizontalLayout_general.setStretch(1, 1)
        self.verticalLayout_2.addLayout(self.horizontalLayout_general)
        self.setCentralWidget(self.centralwidget)
        font = QFont('Times New Roman')
        pointSize = font.pointSize()
        font.setPixelSize(pointSize * 90 / 36)
        self.label_logo.setFont(font)
        self.pushButton_locationLoading.setFont(font)
        self.pushButton_videoLoading.setFont(font)
        self.comboBox.setFont(font)
        self.pushButton_bboxSetting.setFont(font)
        self.pushButton_algorithmProcessing.setFont(font)
        self.label_comboBox.setFont(font)
        self.label_spinBox.setFont(font)
        self.textBrowser.setFont(font)
        self.spinBox.setFont(font)
        self.selectBox.setFont(font)
        self.setFont(font)
        self.label_logo.setPixmap(QPixmap(':rc/logo.png'))
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "SiamBOMB"))
        MainWindow.setWindowIcon(QIcon(':rc/icon.png'))  # Thanks Open Source Site: https://www.iconfont.cn/
        self.progressBar.setFormat(_translate("MainWindow", "STAND BY"))
        self.pushButton_locationLoading.setText(_translate("MainWindow", "&Location Loading"))
        self.pushButton_videoLoading.setText(_translate("MainWindow", "&Video Loading"))
        self.pushButton_outputConfiguring.setText(_translate("MainWindow", "&Output Configuring"))
        self.pushButton_bboxSetting.setText(_translate("MainWindow", "&B-box Setting"))
        self.pushButton_algorithmProcessing.setText(_translate("MainWindow", "&Algorithm Processing"))
        self.label_comboBox.setText('   Model: ')
        self.label_spinBox.setText(' ID: ')
        self.label_selectBox.setText('   Frame: ')
        self.textBrowser.setText('Welcome to SiamBOMB!\n' +
                                 'Copyright © 2020 MiRA,\n' +
                                 'Institute of Automation, CAS.\n' +
                                 'Under the Apache 2.0 license.\n' +
                                 'All rights reserved.')
        # Combo activation
        _model_list = list(json.load(open('./pyqt5/model_list.json', 'r')).keys())
        self.comboBox.addItems(_model_list)
        self.comboBox.setCurrentIndex(-1)
        self.comboBox.setEditText('Choosing One')

    def paintEvent(self, event):
        if self.isPainting:
            if not self.first_frame:  # Reconfiguring
                if self.INIT in self.info_store.keys():
                    for id, bbox in self.info_store[self.INIT].items():
                        x, y, w, h = bbox
                        shape = Rect()
                        shape.start = QPoint(x, y)
                        shape.end = QPoint(x + w, y + h)
                        self.rectList.append(shape)
                self.pp.begin(self.tempPix)
                pen = QPen(Qt.green, 4, Qt.SolidLine)
                self.pp.setPen(pen)
                for shape in self.rectList:
                    shape.paint(self.pp)
                self.pp.end()
                self.label_image.setPixmap(self.tempPix)
            else:
                self.pp.begin(self.tempPix)
                pen = QPen(Qt.green, 4, Qt.SolidLine)
                self.pp.setPen(pen)
                for shape in self.rectList:
                    shape.paint(self.pp)
                self.pp.end()
                self.label_image.setPixmap(self.tempPix)

    def mousePressEvent(self, event):
        if self.isPainting:
            if event.button() == Qt.LeftButton:
                self.shape = Rect()
                if self.shape is not None:
                    self.perm = False
                    self.rectList.append(self.shape)
                    label_left = self.label_image.geometry().left()
                    label_top = self.label_image.geometry().top()
                    label_width = self.label_image.geometry().width()
                    label_height = self.label_image.geometry().height()
                    self.shape.setStart(QPoint(event.pos().x() - (label_width - self.W) // 2 - label_left,
                                               event.pos().y() - (label_height - self.H) // 2 - label_top))
                    self.shape.setEnd(QPoint(event.pos().x() - (label_width - self.W) // 2 - label_left,
                                             event.pos().y() - (label_height - self.H) // 2 - label_top))
                self.update()
            if not self.first_frame:
                if event.button() == Qt.RightButton:
                    if self.INIT not in self.info_store.keys():
                        return
                    label_left = self.label_image.geometry().left()
                    label_top = self.label_image.geometry().top()
                    label_width = self.label_image.geometry().width()
                    label_height = self.label_image.geometry().height()
                    self.perm = False
                    x_pos = event.pos().x() - (label_width - self.W) // 2 - label_left
                    y_pos = event.pos().y() - (label_height - self.H) // 2 - label_top
                    temp_bbox_setting = deepcopy(self.info_store[self.INIT])
                    def info_setting(id):
                        if id not in self.info['sequence_object_ids']:
                            self.info['sequence_object_ids'].append(id)
                        if id in self.info['init_bbox']:
                            self.info['init_bbox'].pop(id)
                        if id in self.info['init_object_ids']:
                            self.info['init_object_ids'].remove(id)
                            self.info['sequence_object_ids'].remove(id)
                    if self.spinBox.value() == 0:
                        for id, bbox in self.info_store[self.INIT].items():
                            x, y, w, h = bbox
                            if (x < x_pos < x+w) and (y < y_pos < y+h):
                                temp_bbox_setting.pop(id)
                                info_setting(id)
                    else:
                        id = self.spinBox.value()
                        try:
                            x, y, w, h = temp_bbox_setting[id]
                            if (x < x_pos < x+w) and (y < y_pos < y+h):
                                temp_bbox_setting.pop(id)
                                info_setting(id)
                        except:
                            print('[INFO] Throwing away this id is already done')
                    self.info_store[self.INIT] = temp_bbox_setting
                    self.update()

    def mouseReleaseEvent(self, event):
        if self.isPainting:
            if event.button() == Qt.LeftButton:
                x = self.shape.startPoint().x()
                y = self.shape.startPoint().y()
                w = self.shape.endPoint().x() - self.shape.startPoint().x()
                h = self.shape.endPoint().y() - self.shape.startPoint().y()
                if w < 0:
                    w = -w
                    x -= w
                if h < 0:
                    h = -h
                    y -= h
                if self.INIT not in self.info_store.keys():
                    self.info_store[self.INIT] = {}
                bbox_setting = self.info_store[self.INIT]
                if (0 <= x < self.W) and (0 <= y < self.H):
                    if self.spinBox.value() == 0:
                        if len(self.info['object_ids']) == 0:
                            bbox_setting[1] = [x, y, w, h]
                            if 1 not in self.info['object_ids']:
                                self.info['object_ids'].append(1)
                            if not self.first_frame:
                                self.info['init_bbox'][1] = [x, y, w, h]
                                if 'init_object_ids' not in self.info.keys() or \
                                    1 not in self.info['init_object_ids']:
                                    self.info['init_object_ids'].append(1)
                        else:
                            id = max(self.info['object_ids']) + 1
                            bbox_setting[id] = [x, y, w, h]
                            self.info['object_ids'].append(id)
                            if not self.first_frame:
                                self.info['init_bbox'][id] = [x, y, w, h]
                                if 'init_object_ids' not in self.info.keys() or \
                                    id not in self.info['init_object_ids']:
                                    self.info['init_object_ids'].append(id)
                    else:
                        id = self.spinBox.value()
                        bbox_setting[id] = [x, y, w, h]
                        if not self.first_frame:
                            self.info['init_bbox'][id] = [x, y, w, h]
                            if id not in self.info['sequence_object_ids']:
                                self.info['sequence_object_ids'].append(id)
                    bbox_setting_text = ''
                    for id, bbox in sorted(list(bbox_setting.items())):
                        bbox_setting_text += '\n' + str(id) + ': ' + str(bbox)
                    self.textBrowser.append('————————————\nB-box Setting Now: '
                                            + bbox_setting_text)
                else:  # Point blank space of mainWin to get the focus
                    self.setFocus()
                self.perm = True
                self.shape = None
                try:
                    for id, bbox in self.info_store[self.INIT].items():
                        x, y, w, h = bbox
                        cv2.putText(self.paint_frame, str(id), (x, y-5), \
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass
                self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                self.update()
            if not self.first_frame:
                if event.button() == Qt.RightButton:
                    self.perm = True
                    self.shape = None
                    self.rectList = []
                    try:
                        for id, bbox in self.info_store[self.INIT].items():
                            x, y, w, h = bbox
                            cv2.putText(self.paint_frame, str(id), (x, y-5), \
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    except:
                        pass
                    self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                    bbox_setting = self.info_store[self.INIT]
                    bbox_setting_text = ''
                    for id, bbox in sorted(list(bbox_setting.items())):
                        bbox_setting_text += '\n' + str(id) + ': ' + str(bbox)
                    self.textBrowser.append('————————————\nB-box Setting Now: '
                                            + bbox_setting_text)
                    self.update()

    def mouseMoveEvent(self, event):
        if self.isPainting:
            self.endPoint = event.pos()
            if event.buttons() & Qt.LeftButton:
                if self.shape is not None and not self.perm:
                    label_left = self.label_image.geometry().left()
                    label_top = self.label_image.geometry().top()
                    label_width = self.label_image.geometry().width()
                    label_height = self.label_image.geometry().height()
                    self.shape.setEnd(QPoint(event.pos().x() - (label_width - self.W) // 2 - label_left,
                                             event.pos().y() - (label_height - self.H) // 2 - label_top))
                    self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                    self.update()

    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Caution', 'Are You Sure to Exit? ' +
                                     '(Note That All Progress Will Not Be Saved.)',
                                     QMessageBox.Yes, QMessageBox.No)
        if reply == QMessageBox.Yes:
            if self.configWin.isEnabled():
                self.configWin.close()
            event.accept()
        else:
            event.ignore()
