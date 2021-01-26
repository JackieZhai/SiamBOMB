# -*- coding: utf-8 -*-

# Created by: PyQt5 UI code generator 5.10.1
#
# Author: JackieZhai @ MiRA, CASIA

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QPixmap, QPen, QFont, QIcon
from PyQt5.QtCore import Qt, QPoint
from copy import deepcopy
import window_rc  # pyrcc5 *.qrc -o *.py


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


class Ui_MainWindow(QWidget):
    def __init__(self, parent=None):
        super(Ui_MainWindow, self).__init__(parent)
        # 流的宽、高、总帧数、初始帧数、当前帧数
        self.W = 800
        self.H = 600
        self.F = 0
        self.INIT = 0
        self.N = 0
        self.setMouseTracking(True)
        self.lastPoint = QPoint()
        self.endPoint = QPoint()
        # 存储b-box坐标信息
        self.bbox_list = []
        # 辅助画布
        self.pp = QPainter()
        self.paint_frame = None
        self.tempPix = QPixmap(self.W, self.H)
        self.tempPix.fill(Qt.white)
        self.shape = None
        self.rectList = []
        self.perm = False
        # 是否处于绘制阶段
        self.isPainting = False
        # 是否处于初始化阶段
        self.first_frame = False
        # 目前状态 Suspending|Location|Video|Camera = 0|1|2|3
        self.isStatus = 0
        self.isAlgorithm = False
        self.isReSet = 0
        self.setupUi()

    def setupUi(self):
        self.setObjectName("MainWindow")
        self.resize(1920, 1080)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHeightForWidth(self.sizePolicy().hasHeightForWidth())
        self.setSizePolicy(sizePolicy)
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
        self.verticalLayout_stream.addWidget(self.progressBar)
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
        self.pushButton_cameraLoading = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.pushButton_cameraLoading.sizePolicy().hasHeightForWidth())
        self.pushButton_cameraLoading.setSizePolicy(sizePolicy)
        self.pushButton_cameraLoading.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.pushButton_cameraLoading.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.pushButton_cameraLoading.setObjectName("pushButton_cameraLoading")
        self.verticalLayout_button.addWidget(self.pushButton_cameraLoading)
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
        self.horizontalLayout_spin = QtWidgets.QHBoxLayout()
        self.horizontalLayout_spin.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_spin.setContentsMargins(-1, 15, -1, -1)
        self.horizontalLayout_spin.setObjectName("horizontalLayout_spin")
        self.spinBox = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.spinBox.sizePolicy().hasHeightForWidth())
        self.spinBox.setSizePolicy(sizePolicy)
        self.spinBox.setObjectName("spinBox")
        self.horizontalLayout_spin.addWidget(self.spinBox)
        self.label_spinBox = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.label_spinBox.sizePolicy().hasHeightForWidth())
        self.label_spinBox.setSizePolicy(sizePolicy)
        self.label_spinBox.setObjectName("label_spinBox")
        self.horizontalLayout_spin.addWidget(self.label_spinBox)
        self.horizontalLayout_spin.setStretch(0, 2)
        self.horizontalLayout_spin.setStretch(1, 8)
        self.verticalLayout_operate.addLayout(self.horizontalLayout_spin)
        self.horizontalLayout_check = QtWidgets.QHBoxLayout()
        self.horizontalLayout_check.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_check.setSpacing(0)
        self.horizontalLayout_check.setObjectName("horizontalLayout_check")
        self.checkBox = QtWidgets.QCheckBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.checkBox.sizePolicy().hasHeightForWidth())
        self.checkBox.setSizePolicy(sizePolicy)
        self.checkBox.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.checkBox.setObjectName("checkBox")
        self.horizontalLayout_check.addWidget(self.checkBox)
        self.selectBox = QtWidgets.QSpinBox(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.selectBox.sizePolicy().hasHeightForWidth())
        self.selectBox.setSizePolicy(sizePolicy)
        self.selectBox.setMaximum(16777215)
        self.selectBox.setObjectName("selectBox")
        self.horizontalLayout_check.addWidget(self.selectBox)
        self.horizontalLayout_check.setStretch(0, 3)
        self.horizontalLayout_check.setStretch(1, 1)
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
        self.scrollBar = QtWidgets.QScrollBar(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHeightForWidth(self.scrollBar.sizePolicy().hasHeightForWidth())
        self.scrollBar.setSizePolicy(sizePolicy)
        self.scrollBar.setOrientation(QtCore.Qt.Horizontal)
        self.scrollBar.setObjectName("scrollBar")
        self.scrollBar.setMaximum(0)
        self.verticalLayout_message.addWidget(self.scrollBar)
        self.verticalLayout_message.setStretch(0, 10)
        self.verticalLayout_message.setStretch(1, 1)
        self.verticalLayout_operate.addLayout(self.verticalLayout_message)
        self.verticalLayout_operate.setStretch(0, 10)
        self.verticalLayout_operate.setStretch(1, 16)
        self.verticalLayout_operate.setStretch(2, 2)
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
        self.progressBar.setFont(font)
        self.label_logo.setFont(font)
        self.pushButton_locationLoading.setFont(font)
        self.pushButton_videoLoading.setFont(font)
        self.pushButton_cameraLoading.setFont(font)
        self.pushButton_bboxSetting.setFont(font)
        self.pushButton_algorithmProcessing.setFont(font)
        self.label_spinBox.setFont(font)
        self.checkBox.setFont(font)
        self.textBrowser.setFont(font)
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
        self.pushButton_cameraLoading.setText(_translate("MainWindow", "&Camera Loading"))
        self.pushButton_bboxSetting.setText(_translate("MainWindow", "&B-box Setting"))
        self.pushButton_algorithmProcessing.setText(_translate("MainWindow", "&Algorithm Processing"))
        self.checkBox.setText(_translate("MainWindow", " &Data Saving    Current: "))
        self.label_spinBox.setText('   Behavior Analysis Selecting')
        self.textBrowser.setText('Welcome to SiamBOMB!\n' +
                                 'Copyright © 2020 MiRA,\n' +
                                 'Institute of Automation, CAS.\n' +
                                 'Under the Apache 2.0 license.\n' +
                                 'All rights reserved.')

    def paintEvent(self, event):
        if self.isPainting and (not self.isReSet):
            self.pp.begin(self.tempPix)
            pen = QPen(Qt.green, 4, Qt.SolidLine)
            self.pp.setPen(pen)
            for shape in self.rectList:
                shape.paint(self.pp)
            self.pp.end()
            self.label_image.setPixmap(self.tempPix)
        elif self.isReSet:
            self.pp.begin(self.tempPix)
            pen = QPen(Qt.green, 4, Qt.SolidLine)
            self.pp.setPen(pen)
            if 0 <= self.INIT < len(self.bbox_list_predict):
                for bbox in self.bbox_list_predict[self.INIT]:
                    x, y, w, h = bbox
                    shape = Rect()
                    shape.start = QPoint(x, y)
                    shape.end = QPoint(x + w, y + h)
                    self.rectList.append(shape)
            for shape in self.rectList:
                shape.paint(self.pp)
            self.pp.end()
            self.label_image.setPixmap(self.tempPix)

    def mousePressEvent(self, event):
        if self.isPainting or self.isReSet:
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
        if self.isReSet:
            if event.button() == Qt.RightButton:
                label_left = self.label_image.geometry().left()
                label_top = self.label_image.geometry().top()
                label_width = self.label_image.geometry().width()
                label_height = self.label_image.geometry().height()
                self.perm = False
                x_pos = event.pos().x() - (label_width - self.W) // 2 - label_left
                y_pos = event.pos().y() - (label_height - self.H) // 2 - label_top
                temp_bbox_list = deepcopy(self.bbox_list_predict[self.INIT])
                for bbox in self.bbox_list_predict[self.INIT]:
                    x, y, w, h = bbox
                    if (x < x_pos < x+w) and (y < y_pos < y+h):
                        temp_bbox_list.remove(bbox)
                self.bbox_list_predict[self.INIT] = temp_bbox_list
                self.update()

    def mouseReleaseEvent(self, event):
        if self.isPainting or self.isReSet:
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
                if self.isReSet:
                    if (0 <= x < self.W) and (0 <= y < self.H):
                        while (len(self.bbox_list_predict) <= self.INIT):
                            self.bbox_list_predict.append([])
                        self.bbox_list_predict[self.INIT].append((x, y, w, h))
                    bbox_setting = self.bbox_list_predict[self.INIT]
                else:
                    if (0 <= x < self.W) and (0 <= y < self.H):
                        self.bbox_list.append((x, y, w, h))
                    bbox_setting = self.bbox_list
                self.perm = True
                self.shape = None
                bbox_setting_text = ''
                for item in bbox_setting:
                    bbox_setting_text += '\n' + str(item)
                self.textBrowser.append('————————————\nB-box Setting Now: '
                                        + bbox_setting_text)
                self.update()
        if self.isReSet:
            if event.button() == Qt.RightButton:
                self.perm = True
                self.shape = None
                self.rectList = []
                self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                bbox_setting = self.bbox_list_predict[self.INIT]
                bbox_setting_text = ''
                for item in bbox_setting:
                    bbox_setting_text += '\n' + str(item)
                self.textBrowser.append('————————————\nB-box Setting Now: '
                                        + bbox_setting_text)
                self.update()

    def mouseMoveEvent(self, event):
        if self.isPainting or self.isReSet:
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
