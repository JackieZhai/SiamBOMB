# -*- coding: utf-8 -*-

# Author: JackieZhai @ MiRA, CASIA

import sys
from os import system, path, listdir
from copy import deepcopy
from glob import glob
from imutils.video import FPS
import numpy as np
import cv2
import torch

from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from window_ui import Ui_MainWindow
from PyQt5.QtGui import QPixmap, QImage, QCursor
from PyQt5.QtCore import Qt

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

torch.set_num_threads(1)


class MyMainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.isDracula = False
        # Connect the on-clicked functions
        self.pushButton_locationLoading.clicked.connect(self.location_loading)
        self.pushButton_videoLoading.clicked.connect(self.video_loading)
        self.pushButton_cameraLoading.clicked.connect(self.camera_loading)
        self.pushButton_bboxSetting.clicked.connect(self.bbox_setting)
        self.pushButton_algorithmProcessing.clicked.connect(self.algorithm_processing)
        self.scrollBar.valueChanged.connect(self.slider_change)
        self.selectBox.valueChanged.connect(self.select_change)
        self.checkBox.stateChanged.connect(self.checkbox_change)
        # Message box ignore
        self.bbox_tips = True
        self.save_tips = True
        # Initialize trackers
        model_location = './pysot/experiments/siammaske_r50_l3'
        self.config = model_location + '/config.yaml'
        self.snapshot = model_location + '/model.pth'
        self.tracker_name = model_location.split('/')[-1]
        self.video_name = ''
        cfg.merge_from_file(self.config)
        cfg.CUDA = torch.cuda.is_available()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        model = ModelBuilder()
        model.load_state_dict(torch.load(self.snapshot, map_location=lambda storage, loc: storage.cpu()))
        model.eval().to(device)
        self.tracker = build_tracker(model)
        self.vs = None
        self.analysis_box = None
        self.analysis_max = 10
        self.save_location = ''
        self.afterCamera = False
        self.bbox_list_predict = []  # [time][tracker]

    def cv2_to_qimge(self, cvImg):
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg

    def get_frame(self, video_name, frame_num):
        if not video_name:
            return
        elif video_name.endswith('avi') or \
                video_name.endswith('mp4'):
            cap = cv2.VideoCapture(video_name)
            cap.set(1, frame_num)
            ret, frame = cap.read()
            if ret:
                return frame
            else:
                return
        else:
            images = glob(path.join(video_name, '*.jp*'))
            images = sorted(images,
                            key=lambda x: int(x.split('\\')[-1].split('.')[0]))
            if 0 <= frame_num < len(images):
                frame = cv2.imread(images[frame_num])
                return frame
            else:
                return


    def get_frames(self, video_name):
        if not video_name:
            return
        elif video_name.endswith('avi') or \
                video_name.endswith('mp4'):
            cap = cv2.VideoCapture(video_name)
            self.W = cap.get(3)
            self.H = cap.get(4)
            self.F = int(cap.get(7))
            self.scrollBar.setMaximum(self.F - 1)
            cap.set(1, self.INIT)
            while True:
                ret, frame = cap.read()
                if ret:
                    yield frame
                else:
                    break
        else:
            images = glob(path.join(video_name, '*.jp*'))
            images = sorted(images,
                            key=lambda x: int(x.split('\\')[-1].split('.')[0]))
            self.F = len(images)
            self.scrollBar.setMaximum(self.F - 1)
            for img in range(self.INIT, self.F):
                frame = cv2.imread(images[img])
                self.W = frame.shape[1]
                self.H = frame.shape[0]
                yield frame

    def analysis_init(self):
        self.analysis_box = []
        for i in range(len(self.bbox_list)):
            q_trans = []
            q_trans_loc = 0
            q_segmove = []
            q_segmove_loc = 0
            for j in range(self.analysis_max):
                q_trans.append(None)
                q_segmove.append(None)
            pre_center = None
            pre_mask = None
            self.analysis_box.append([q_trans, q_trans_loc, q_segmove, q_segmove_loc, pre_center, pre_mask])

    def behavior_analysis(self, frame, b, center, mask):
        if self.analysis_box[b][4] is None:
            self.analysis_box[b][0][self.analysis_box[b][1]] = center
        else:
            self.analysis_box[b][0][self.analysis_box[b][1]] = (center[0] - self.analysis_box[b][4][0],
                                                                center[1] - self.analysis_box[b][4][1])
        self.analysis_box[b][1] += 1
        if self.analysis_box[b][1] >= self.analysis_max:
            self.analysis_box[b][1] = 0
        mean_trans = 0.0
        for item in self.analysis_box[b][0]:
            if item is not None:
                mean_trans += np.sqrt(item[0] * item[0] + item[1] * item[1])
        mean_trans /= self.analysis_max

        if self.analysis_box[b][4] is None:
            self.analysis_box[b][2][self.analysis_box[b][3]] = (mask, mask)
        else:
            self.analysis_box[b][2][self.analysis_box[b][3]] = (np.bitwise_and(mask, self.analysis_box[b][5]),
                                                                np.bitwise_or(mask, self.analysis_box[b][5]))
        self.analysis_box[b][3] += 1
        if self.analysis_box[b][3] >= self.analysis_max:
            self.analysis_box[b][3] = 0
        mean_segmove = 0.0
        for item in self.analysis_box[b][2]:
            if item is not None:
                iou = np.sum(item[0]) / np.sum(item[1])
                mean_segmove += 1 - iou
        mean_segmove /= self.analysis_max

        if mean_trans > 10:
            mean_state_text = "State: LM(Locomotion)"
        else:
            if mean_segmove > 0.1:
                mean_state_text = "State: NM(Non-locomotor movement)"
            else:
                mean_state_text = "State: KS(Keep Stillness)"

        if b == self.spinBox.value():
            text = "Trans: {:.2f} pixel/frame".format(mean_trans)
            cv2.putText(frame, text, (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            text = "SegMove: {:.2f} %/frame".format(mean_segmove * 100)
            cv2.putText(frame, text, (60, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            cv2.putText(frame, mean_state_text, (60, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # text = "Trans: {:.2f} pixel/frame".format(mean_trans)
            # cv2.putText(frame, text, (360, 370), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            # text = "SegMove: {:.2f} %/frame".format(mean_segmove * 100)
            # cv2.putText(frame, text, (360, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            # cv2.putText(frame, mean_state_text, (360, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        self.analysis_box[b][4] = center
        self.analysis_box[b][5] = mask

    def location_loading(self):
        if self.isStatus == 1:
            self.isStatus = 0
            self.pushButton_bboxSetting.setText("&B-box Setting")
            return
        elif self.isStatus != 0:
            print('[INFO] Error in interrupting algorithm.')
            QMessageBox.information(self, 'Warning', 'You Must Stop Algorithm Process First. ' +
                                    '(Key: ... Suspending)', QMessageBox.Ok)
            return
        self.afterCamera = False
        self.video_name = QFileDialog.getExistingDirectory(self, 'Choose Frames Location', './')
        if self.video_name == '':
            return
        self.textBrowser.append('————————————\nStream Path: \n' + self.video_name)
        is_frame = False
        self.INIT = 0
        self.N = 0
        self.scrollBar.setProperty('value', 0)
        for frame in self.get_frames(self.video_name):
            is_frame = True
            self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
            self.paint_frame = frame
            QApplication.processEvents()
            break
        if not is_frame:
            print('[INFO] Error in location selecting.')
            QMessageBox.information(self, 'Warning', 'You Must Choose Some Location (That Contains Images). ' +
                                    '(Shortcut Key: Alt + L)', QMessageBox.Ok)
            return
        self.textBrowser.append('————————————\nTotal Frames: ' + str(self.F))
        self.bbox_list_predict = []
        self.isStatus = 1

    def video_loading(self):
        if self.isStatus == 2:
            self.isStatus = 0
            self.pushButton_bboxSetting.setText("&B-box Setting")
            return
        elif self.isStatus != 0:
            print('[INFO] Error in interrupting algorithm.')
            QMessageBox.information(self, 'Warning', 'You Must Stop Algorithm Process First. ' +
                                    '(Key: ... Suspending)', QMessageBox.Ok)
            return
        self.afterCamera = False
        self.video_name = QFileDialog.getOpenFileName(self, 'Choose Frames File', './', 'Video file (*.avi *.mp4)')
        self.video_name, _ = self.video_name
        if self.video_name == '':
            return
        self.textBrowser.append('————————————\nStream Path: \n' + self.video_name)
        is_frame = False
        self.INIT = 0
        self.N = 0
        self.scrollBar.setProperty('value', 0)
        for frame in self.get_frames(self.video_name):
            is_frame = True
            self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
            self.paint_frame = frame
            QApplication.processEvents()
            break
        if not is_frame:
            print('[INFO] Error in video reading.')
            QMessageBox.information(self, 'Warning', 'You Must Re-choose Some Video (That Could Be Read). ' +
                                    '(Shortcut Key: Alt + V)', QMessageBox.Ok)
            return
        self.textBrowser.append('————————————\nTotal Frames: ' + str(self.F))
        self.bbox_list_predict = []
        self.isStatus = 2

    def camera_loading(self):
        if self.isStatus == 1 or self.isStatus == 2:
            print('[INFO] Error in interrupting algorithm.')
            QMessageBox.information(self, 'Warning', 'You Must Stop Algorithm Process First. ' +
                                    '(Key: ... Suspending)', QMessageBox.Ok)
            return
        if self.isStatus == 3:
            self.isStatus = 0
            print("[INFO] Exporting webcam stream.")
            self.pushButton_cameraLoading.setText('&Camera Loading')
            self.bbox_list = []
            self.rectList = []
            self.paint_frame = None
            if self.vs is not None:
                self.vs.release()
            self.isPainting = False
            self.afterCamera = True
            self.F = len(listdir(self.save_location))
            self.scrollBar.setMaximum(self.F)
            self.scrollBar.setProperty('value', self.F)
        else:
            self.isStatus = 3
            self.pushButton_cameraLoading.setText('&Camera Suspending')
            self.afterCamera = False
        if self.isStatus == 3:
            trackers = []
            mirror = True
            print("[INFO] Importing webcam stream.")
            self.vs = cv2.VideoCapture(0)
            label_width = self.label_image.geometry().width()
            label_height = self.label_image.geometry().height()
            self.vs.set(3, label_width)
            self.vs.set(4, label_height)
            self.W = self.vs.get(3)
            self.H = self.vs.get(4)
            fps_cal = None
            self.N = 0
            self.first_frame = True
            save_loc_d = './ans/' + '__webcam__'
            save_loc_t = save_loc_d + '/' + str(self.tracker_name)
            self.save_location = save_loc_t
            system('mkdir ' + save_loc_t.replace('/', '\\'))
            system('del /q ' + save_loc_t.replace('/', '\\'))
            while True:
                _, frame = self.vs.read()
                if mirror:
                    frame = cv2.flip(frame, 1)
                self.paint_frame = frame
                if (not self.isPainting) and len(self.bbox_list):
                    self.progressBar.setFormat('HANDLE')
                    if self.first_frame:
                        print('[INFO] Here are initialization of processing webcam.')
                        for b in range(len(self.bbox_list)):
                            trackers.append(deepcopy(self.tracker))
                        for b in range(len(self.bbox_list)):
                            trackers[b].init(frame, self.bbox_list[b])
                        fps_cal = FPS().start()
                        self.analysis_init()
                        self.first_frame = False
                    else:
                        masks = None
                        for b in range(len(self.bbox_list)):
                            outputs = trackers[b].track(frame)
                            if 'polygon' in outputs:
                                polygon = np.array(outputs['polygon']).astype(np.int32)
                                cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                              True, (0, 255, 0), 3)
                                mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * 255)
                                mask = mask.astype(np.uint8)
                                if masks is None:
                                    masks = mask
                                else:
                                    masks += mask
                                polygon_xmean = (polygon[0] + polygon[2] + polygon[4] + polygon[6]) / 4
                                polygon_ymean = (polygon[1] + polygon[3] + polygon[5] + polygon[7]) / 4
                                cv2.rectangle(frame, (int(polygon_xmean) - 1, int(polygon_ymean) - 1),
                                              (int(polygon_xmean) + 1, int(polygon_ymean) + 1), (0, 255, 0), 3)
                                self.behavior_analysis(frame, b,
                                                       (polygon_xmean, polygon_ymean), (mask > 0))
                            else:
                                bbox = list(map(int, outputs['bbox']))
                                cv2.rectangle(frame, (bbox[0], bbox[1]),
                                              (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 3)
                        frame[:, :, 2] = (masks > 0) * 255 * 0.75 + (masks == 0) * frame[:, :, 2]
                        fps_cal.update()
                        fps_cal.stop()
                        text = "FPS: {:.2f}".format(fps_cal.fps())
                        cv2.putText(frame, text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    # 非处理中，延迟取帧：0.05s
                    if (self.isPainting):
                        self.progressBar.setFormat('DRAW')
                    else:
                        self.progressBar.setFormat('PHOTOGRAPHY')
                    cv2.waitKey(50)
                self.N += 1
                if self.checkBox.checkState():
                    save_loc_i = save_loc_t + "/" + str(self.N).zfill(4) + ".jpg"
                    cv2.imwrite(save_loc_i, frame)
                self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
                QApplication.processEvents()
                if self.isStatus != 3:
                    self.vs.release()
                    self.progressBar.setFormat('STAND BY')
                    QApplication.processEvents()
                    break

    def bbox_setting(self):
        if self.isAlgorithm:
            self.progressBar.setFormat('RE-DRAW')
            self.pushButton_bboxSetting.setText("&B-box Reconfiguring")
            self.isReSet = self.isStatus
            self.isStatus = 0
            self.isAlgorithm = False
            self.INIT = self.N
            self.rectList = []
            self.label_image.setCursor(QCursor(Qt.CrossCursor))
            self.paint_frame = self.get_frame(self.video_name, self.N)
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
            return
        if self.isPainting or self.isReSet:
            if self.isReSet and (len(self.bbox_list_predict) > self.INIT):
                print('[INFO] Throw away the setting bbox of: ' + str(self.bbox_list_predict[self.INIT]))
                self.bbox_list_predict[self.INIT] = []
            else:
                print('[INFO] Throw away the setting bbox of: ' + str(self.bbox_list))
            self.bbox_list = []
            self.rectList = []
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
        else:
            self.progressBar.setFormat('DRAW')
            self.label_image.setCursor(QCursor(Qt.CrossCursor))
            if self.bbox_tips:
                if self.isStatus is not 3:
                    reply = QMessageBox.information(self, 'Tips', 'You Can Set Initial Frame By: ' +
                                                    'Keyboard \",\" - Previous Frame; ' +
                                                    'Keyboard \".\" - Next Frame. ' +
                                                    '(If You Need)', QMessageBox.Ignore, QMessageBox.Ok)
                    if reply == QMessageBox.Ignore:
                        self.bbox_tips = False
            if self.paint_frame is None:
                QMessageBox.information(self, 'Warning', 'You Must Get Data First. (Alt + L / V / C)',
                                        QMessageBox.Ok)
            else:
                self.isPainting = True
                self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                self.pushButton_bboxSetting.setText("&B-box Reconfiguring")

    def algorithm_processing(self):
        if self.isAlgorithm:
            return
        if self.isReSet:
            self.bbox_list = self.bbox_list_predict[self.INIT]
            self.isStatus = self.isReSet
            self.isReSet = False
        if len(self.bbox_list) == 0:
            print('[INFO] Error in b-box choosing.')
            QMessageBox.information(self, 'Warning', 'You Must Confirm B-box First. (Shortcut Key: Alt + B)',
                                    QMessageBox.Ok)
            return
        self.isPainting = False
        self.pushButton_bboxSetting.setText("&B-box Setting")
        if self.isStatus == 1:
            self.pushButton_locationLoading.setText('&Stream Suspending')
        elif self.isStatus == 2:
            self.pushButton_videoLoading.setText('&Stream Suspending')
        elif self.isStatus == 3:
            self.first_frame = True
            return
        self.label_image.setCursor(QCursor(Qt.ArrowCursor))
        self.isAlgorithm = True
        trackers = []
        self.progressBar.setFormat('INITIALIZE')
        for b in range(len(self.bbox_list)):
            trackers.append(deepcopy(self.tracker))
        print("[INFO] Starting pictures stream.")
        fps_cal = None
        self.first_frame = True
        save_loc_d = './ans/' + self.video_name.split('/')[-1]
        save_loc_t = save_loc_d + '/' + str(self.tracker_name)
        self.save_location = save_loc_t
        self.N = self.INIT
        for frame in self.get_frames(self.video_name):
            if self.first_frame:
                for b in range(len(self.bbox_list)):
                    trackers[b].init(frame, self.bbox_list[b])
                while(len(self.bbox_list_predict) < self.INIT):
                    self.bbox_list_predict.append([])
                if len(self.bbox_list_predict) > self.N:
                    self.bbox_list_predict[self.N] = self.bbox_list
                else:
                    self.bbox_list_predict.append(self.bbox_list)
                self.first_frame = False
                if self.checkBox.checkState():
                    system('mkdir ' + save_loc_t.replace('/', '\\'))
                    system('mkdir ' + (save_loc_t+'/mask').replace('/', '\\'))
                fps_cal = FPS().start()
                self.analysis_init()
                self.pushButton_bboxSetting.setText("&B-box Pausing")
                self.textBrowser.append('————————————\nTotal Frames: ' + str(self.F))
            else:
                if self.isStatus == 0:
                    break
                masks = None
                self.N += 1
                if len(self.bbox_list_predict) > self.N:
                    self.bbox_list_predict[self.N] = []
                else:
                    self.bbox_list_predict.append([])
                for b in range(len(self.bbox_list)):
                    outputs = trackers[b].track(frame)
                    assert 'polygon' in outputs
                    polygon = np.array(outputs['polygon']).astype(np.int32)
                    cv2.polylines(frame, [polygon.reshape((-1, 1, 2))],
                                  True, (0, 255, 0), 2)
                    # mask label should be (255 - object selecting number)?
                    mask = ((outputs['mask'] > cfg.TRACK.MASK_THERSHOLD) * (255 - b))
                    mask = mask.astype(np.uint8)
                    if masks is None:
                        masks = mask
                    else:
                        masks += mask
                    polygon_xmean = (polygon[0] + polygon[2] + polygon[4] + polygon[6]) / 4
                    polygon_ymean = (polygon[1] + polygon[3] + polygon[5] + polygon[7]) / 4
                    cv2.rectangle(frame, (int(polygon_xmean) - 1, int(polygon_ymean) - 1),
                                  (int(polygon_xmean) + 1, int(polygon_ymean) + 1), (0, 255, 0), 2)
                    # self.behavior_analysis(frame, b,
                    #                        (polygon_xmean, polygon_ymean), (mask > 0))

                    bbox = tuple(list(map(int, outputs['bbox'])))
                    self.bbox_list_predict[self.N].append(bbox)
                frame[:, :, 2] = (masks > 0) * 255 * 0.75 + (masks == 0) * frame[:, :, 2]
                fps_cal.update()
                fps_cal.stop()
                text = "FPS: {:.2f}".format(fps_cal.fps())
                cv2.putText(frame, text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                # cv2.putText(frame, text, (360, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                if self.checkBox.checkState():
                    save_loc_i = save_loc_t + "/" + str(self.N).zfill(4) + ".jpg"
                    cv2.imwrite(save_loc_i, frame)
                    save_loc_m = save_loc_t + "/mask/" + str(self.N).zfill(4) + ".jpg"
                    cv2.imwrite(save_loc_m, mask)
                self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
                QApplication.processEvents()
            self.progressBar.setProperty('value', (self.N - self.INIT) * 100 / (self.F - 1 - self.INIT))
            self.progressBar.setFormat('HANDLE: %p% [{:d}/{:d}]'.format(self.N, self.F - 1))
            self.scrollBar.setProperty('value', self.N)
            QApplication.processEvents()

        self.bbox_list = []
        self.rectList = []
        if not self.isReSet:
            self.paint_frame = None
        del trackers, fps_cal
        print("[INFO] Ending pictures stream.")
        if self.checkBox.checkState():
            # save trackless data
            # jpg_list = glob(save_loc_t + '/*.jpg')
            jpg_num_list = []
            for j in range(len(self.bbox_list_predict)):
                if len(self.bbox_list_predict[j]):
                    jpg_num_list.append(j)
            jpg_num_list.sort()
            last_j = 0
            for i, now_j in enumerate(jpg_num_list):
                self.progressBar.setProperty('value', i * 100 / (len(jpg_num_list) - 1))
                self.progressBar.setFormat('Pausing: %p%')
                QApplication.processEvents()
                for j in range(last_j, now_j):
                    frame = self.get_frame(self.video_name, j)
                    save_loc_i = save_loc_t + "/" + str(j).zfill(4) + ".jpg"
                    cv2.imwrite(save_loc_i, frame)
                    save_loc_m = save_loc_t + "/mask/" + str(j).zfill(4) + ".jpg"
                    cv2.imwrite(save_loc_m, np.zeros((frame.shape[0], frame.shape[1])))
                last_j = now_j + 1
            # save bbox data
            with open(save_loc_t + '/bbox.txt', 'w') as f:
                for item in self.bbox_list_predict:
                    f.write(str(item) + '\n')
            np.save(save_loc_t + '/bbox.npy', np.array(self.bbox_list_predict))
            # save to a video
            if last_j == self.F:
                fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
                if self.isStatus == 1:
                    frame_0 = self.get_frame(self.video_name, 0)
                    wri = cv2.VideoWriter(save_loc_t + '/video.avi', fourcc,
                                          30, (frame_0.shape[1], frame.shape[0]))
                elif self.isStatus == 2:
                    cap = cv2.VideoCapture(self.video_name)
                    wri = cv2.VideoWriter(save_loc_t + '/video.avi', fourcc,
                                          int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))
                for j in range(last_j):
                    frame = cv2.imread(save_loc_t + '/' + str(j).zfill(4) + ".jpg")
                    wri.write(frame)
                    self.progressBar.setProperty('value', j * 100 / (last_j - 1))
                    self.progressBar.setFormat('SAVE: %p% [{:d}/{:d}]'.format(j, last_j - 1))
                    QApplication.processEvents()
                wri.release()
                self.pushButton_bboxSetting.setText('&B-box Setting')
                QApplication.processEvents()

        self.isStatus = 0
        self.pushButton_locationLoading.setText('&Location Loading')
        self.pushButton_videoLoading.setText('&Video Loading')
        self.progressBar.setFormat('STAND BY')
        self.progressBar.setProperty('value', 0)
        # self.F = f + 1 + self.INIT
        # self.scrollBar.setMaximum(f + self.INIT)
        self.isAlgorithm = False


    def keyPressEvent(self, event):
        if self.isPainting or self.isReSet:
            if event.key() == Qt.Key_Period:
                if self.scrollBar.value() < self.F - 1:
                    self.selectBox.setProperty('value', self.INIT + 1)
                    self.perm = True
            elif event.key() == Qt.Key_Comma:
                if self.scrollBar.value() > 0:
                    self.selectBox.setProperty('value', self.INIT - 1)
                    self.perm = True
        elif self.afterCamera and self.checkBox.checkState():
            if event.key() == Qt.Key_Period:
                if self.scrollBar.value() < self.F:
                    self.scrollBar.setProperty('value', self.scrollBar.value() + 1)
            elif event.key() == Qt.Key_Comma:
                if self.scrollBar.value() > 1:
                    self.scrollBar.setProperty('value', self.scrollBar.value() - 1)
        elif (self.isStatus is 0) and self.checkBox.checkState():
            if event.key() == Qt.Key_Period:
                if self.scrollBar.value() < self.F - 1:
                    self.scrollBar.setProperty('value', self.scrollBar.value() + 1)
            elif event.key() == Qt.Key_Comma:
                if self.scrollBar.value() > self.INIT + 1:
                    self.scrollBar.setProperty('value', self.scrollBar.value() - 1)

    def select_change(self):
        if self.isPainting or self.isReSet:
            self.scrollBar.setProperty('value', self.selectBox.value())

    def slider_change(self):
        self.selectBox.setProperty('value', self.scrollBar.value())
        if self.isPainting or self.isReSet:
            self.INIT = self.scrollBar.value()
            self.rectList = []
            frame = self.get_frame(self.video_name, self.INIT)
            self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
            self.paint_frame = frame
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
            QApplication.processEvents()
            # self.textBrowser.append('————————————\nFirst Frame Now: ' + str(self.INIT) +
            #                         ' of ' + str(self.F))
        elif self.afterCamera and self.checkBox.checkState():
            if self.scrollBar.value() < 1:
                self.scrollBar.setProperty('value', 1)
            else:
                save_loc_i = self.save_location + '/' + str(self.scrollBar.value()).zfill(4) + ".jpg"
                frame = cv2.imread(save_loc_i)
                self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
                self.paint_frame = frame
                self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                QApplication.processEvents()
        elif (self.isStatus is 0) and self.checkBox.checkState():
            if self.scrollBar.value() <= self.INIT:
                self.scrollBar.setProperty('value', self.INIT + 1)
            else:
                save_loc_i = self.save_location + '/' + str(self.scrollBar.value() - self.INIT).zfill(4) + ".jpg"
                frame = cv2.imread(save_loc_i)
                self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
                self.paint_frame = frame
                self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
                QApplication.processEvents()

    def checkbox_change(self):
        if self.isAlgorithm or (self.isStatus is 3):
            if self.checkBox.checkState():
                self.checkBox.setChecked(False)
            else:
                self.checkBox.setChecked(True)
            QMessageBox.information(self, 'Warning', 'You Must Change Checkbox Before ' +
                                    '/ After the Algorithm Procedure.', QMessageBox.Ok)
            QApplication.processEvents()
        if self.checkBox.checkState():
            if self.save_tips:
                reply = QMessageBox.information(self, 'Tips', 'You Can View Saved Answer By: ' +
                                                'Keyboard \",\" - Previous Frame; ' +
                                                'Keyboard \".\" - Next Frame. ' +
                                                '(After Algorithm Procedure)', QMessageBox.Ignore, QMessageBox.Ok)
                if reply == QMessageBox.Ignore:
                    self.save_tips = False


if __name__ == '__main__':
    app = QApplication(sys.argv)
    myWin = MyMainWindow()

    if myWin.isDracula:
        # Open Source from https://github.com/Kojoley/qss-dracula
        with open('dracula.qss', 'r') as f:
            qssStyle = f.read()
        myWin.setStyleSheet(qssStyle)

    myWin.show()
    sys.exit(app.exec_())