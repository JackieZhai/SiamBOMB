# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 

from os import path
import time
import importlib
from copy import deepcopy
from glob import glob
from imutils.video import FPS
import numpy as np
import cv2
from numpy.lib.npyio import loads
import torch
from collections import OrderedDict

from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap, QCursor
from PyQt5.QtCore import Qt, pyqtSignal, QThread

from .window_base import BaseWindow
from .window_configure import ConfigWindow
from ..pytracking.utils.plotting import overlay_mask

checkBoxcheckState = False


class ComboBoxThread(QThread):
    def __init__(self, mainWin, progressSignal, parent=None):
        super().__init__(parent=parent)
        self.mainWin = mainWin
        self.progressSignal = progressSignal
    
    def _pysot_init(self, config, snapshot):
        # Import depending packages
        from ..pysot.core.config import cfg
        from ..pysot.models.model_builder import ModelBuilder
        from ..pysot.tracker.tracker_builder import build_tracker
        from ..pysot.tracker.multiple_tracker import MultiTracker
        cfg.merge_from_file(config)
        cfg.CUDA = torch.cuda.is_available()
        device = torch.device('cuda' if cfg.CUDA else 'cpu')
        self.progressSignal.emit(20)
        model = ModelBuilder()
        self.progressSignal.emit(40)
        model.load_state_dict(torch.load(snapshot, map_location=lambda storage, loc: storage.cpu()))
        self.progressSignal.emit(60)
        model.eval().to(device)
        self.progressSignal.emit(80)
        self.mainWin.tracker = MultiTracker(build_tracker(model), cfg)
    
    def _pytracking_init(self, tracker_name, parameter_name):
        # Import depending packages
        from ..pytracking.evaluation.multi_object_wrapper import MultiObjectWrapper
        root_abspath = path.dirname(path.dirname(path.abspath(__file__)))
        tracker_module_abspath = path.join(root_abspath, 'pytracking', 'tracker', tracker_name)
        if path.isdir(tracker_module_abspath):
            tracker_loc = locals()
            exec('from ..pytracking.tracker import {} as tracker_module'.format(tracker_name))
            tracker_module = tracker_loc['tracker_module']
            tracker_class = tracker_module.get_tracker_class()
        else:
            tracker_class = None
        param_loc = locals()
        exec('from ..pytracking.parameter.{} import {} as param_module'.format(tracker_name, parameter_name))
        param_module = param_loc['param_module']
        self.progressSignal.emit(20)
        params = param_module.parameters()
        self.progressSignal.emit(40)
        params.tracker_name = tracker_name
        self.progressSignal.emit(60)
        params.param_name = parameter_name
        self.progressSignal.emit(80)
        multiobj_mode = getattr(params, 'multiobj_mode', getattr(tracker_class, 'multiobj_mode', 'default'))
        if multiobj_mode == 'default':
            self.mainWin.tracker = tracker_class(params)
            if hasattr(self.mainWin.tracker, 'initialize_features'):
                self.mainWin.tracker.initialize_features()
        elif multiobj_mode == 'parallel':
            self.mainWin.tracker = MultiObjectWrapper(tracker_class, params, fast_load=True)
        else:
            raise ValueError('Unknown multi object mode {}'.format(multiobj_mode))

    def run(self):
        self.progressSignal.emit(0)
        # Initialize trackers
        if self.mainWin.model_name == 'SiamMask_E':
            self.progressSignal.emit(10)
            model_location = path.join(path.dirname(path.dirname(path.abspath(__file__))), \
                'pysot/experiments/siammaske_r50_l3')
            config = path.join(model_location, 'config.yaml')
            snapshot = path.join(model_location, 'model.pth')
            self._pysot_init(config, snapshot)
            self.progressSignal.emit(100)
        elif self.mainWin.model_name == 'KYS':
            self.progressSignal.emit(10)
            tracker_name = 'kys'
            parameter_name = 'default'
            self._pytracking_init(tracker_name, parameter_name)
            self.progressSignal.emit(100)
        elif self.mainWin.model_name == 'LWL':
            self.progressSignal.emit(10)
            tracker_name = 'lwl'
            parameter_name = 'lwl_boxinit'
            self._pytracking_init(tracker_name, parameter_name)
            self.progressSignal.emit(100)
        elif self.mainWin.model_name == 'KeepTrack':
            self.progressSignal.emit(10)
            tracker_name = 'keep_track'
            parameter_name = 'default_fast'
            self._pytracking_init(tracker_name, parameter_name)
            self.progressSignal.emit(100)
        else:
            self.progressSignal.emit(-1)
            return
        self.progressSignal.emit(101)


class ScrollBarThread(QThread):
    def __init__(self, now_value, main_win, parent=None):
        super().__init__(parent=parent)
        self.now_value = now_value
        self.main_win = main_win

    def run(self):
        self.now_value = self.main_win.scrollBar.value()
        time.sleep(0.01)
        if self.now_value == self.main_win.scrollBar.value():
            self.main_win.scrollBar_trueSignal.emit()


class SiamBOMBWindow(BaseWindow):
    # Define true singals of scrollBar
    scrollBar_trueSignal = pyqtSignal()
    progress_childSignal = pyqtSignal(int)
    def __init__(self, parent=None):
        super(SiamBOMBWindow, self).__init__(parent)
        # Connect the on-clicked functions
        self.comboBox.currentIndexChanged.connect(self.combo_change)
        self.pushButton_locationLoading.clicked.connect(self.location_loading)
        self.pushButton_videoLoading.clicked.connect(self.video_loading)
        self.pushButton_outputConfiguring.clicked.connect(self.output_configuring)
        self.pushButton_bboxSetting.clicked.connect(self.bbox_setting)
        self.pushButton_algorithmProcessing.clicked.connect(self.algorithm_processing)
        self.scrollBar.valueChanged.connect(self.slider_change)
        self.scrollBar_trueSignal.connect(self.slider_true_change)
        self.progress_childSignal.connect(self.progress_child_change)
        self.spinBox.valueChanged.connect(self.spin_change)
        self.selectBox.valueChanged.connect(self.select_change)
        # Configure window initialization
        self.configWin = ConfigWindow(self)
        self.configWin.setEnabled(False)
        # Message box ignore
        self.bbox_tips = True
        self.save_tips = True
        # Ban buttons
        self.pushButton_locationLoading.setEnabled(False)
        self.pushButton_videoLoading.setEnabled(False)
        self.pushButton_outputConfiguring.setEnabled(False)
        self.pushButton_bboxSetting.setEnabled(False)
        self.pushButton_algorithmProcessing.setEnabled(False)
        self.spinBox.setEnabled(False)
        self.selectBox.setEnabled(False)
        self.scrollBar.setEnabled(False)
        # Some initialization
        self.prev_output = None
        self.model_name = ''
        self.video_name = ''
        self.video_activate = True
        self.video_fps = False
        self.video_factor = 1.00
        self.analysis_box = None
        self.analysis_max = 10
        self.save_location = ''

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
                            key=lambda x: int(x.split('/')[-1].split('.')[0]))
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
        if self.pushButton_locationLoading.text().endswith('Suspending'):
            self.pushButton_bboxSetting.setText("&B-box Setting")
            self.pushButton_locationLoading.setText('&Location Loading')
        elif self.pushButton_locationLoading.text().endswith('Loading'):
            self.video_name = QFileDialog.getExistingDirectory(self, 'Choose Frames Location', './')
            if self.video_name == '':
                return
            self.textBrowser.append('————————————\nLoaded Stream: \n' + self.video_name)
            is_frame = False
            self.INIT = 0
            self.N = 0
            self.scrollBar.setProperty('value', 0)
            self.isPainting = False
            self.rectList = []
            self.pushButton_bboxSetting.setText("&B-box Setting")
            self.progressBar.setFormat('STAND BY DATA')
            self.progressBar.setProperty('value', 0)
            for frame in self.get_frames(self.video_name):
                is_frame = True
                frame = self.cv2_to_factor(frame, self.video_factor)
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
            self.bbox_list = []
            self.info = {}
            self.bbox_list_predict = []
            self.info_store = OrderedDict()
            self.first_frame = True
            self.selectBox.setEnabled(True)
            self.scrollBar.setEnabled(True)
            self.pushButton_bboxSetting.setEnabled(True)
        else:
            raise Exception

    def video_loading(self):
        if self.pushButton_videoLoading.text().endswith('Suspending'):
            self.pushButton_bboxSetting.setText("&B-box Setting")
            self.pushButton_videoLoading.setText('&Video Loading')
        elif self.pushButton_videoLoading.text().endswith('Loading'):
            self.video_name = QFileDialog.getOpenFileName(self, 'Choose Frames File', './', 'Video file (*.avi *.mp4)')
            self.video_name, _ = self.video_name
            if self.video_name == '':
                return
            self.textBrowser.append('————————————\nLoaded Stream: \n' + self.video_name)
            is_frame = False
            self.INIT = 0
            self.N = 0
            self.scrollBar.setProperty('value', 0)
            self.isPainting = False
            self.rectList = []
            self.pushButton_bboxSetting.setText("&B-box Setting")
            self.progressBar.setFormat('STAND BY DATA')
            self.progressBar.setProperty('value', 0)
            for frame in self.get_frames(self.video_name):
                is_frame = True
                frame = self.cv2_to_factor(frame, self.video_factor)
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
            self.info = {}
            self.info_store = OrderedDict()
            self.first_frame = True
            self.selectBox.setEnabled(True)
            self.scrollBar.setEnabled(True)
            self.pushButton_bboxSetting.setEnabled(True)
        else:
            raise Exception

    def output_configuring(self):
        self.configWin.show()
        self.setEnabled(False)
        self.configWin.setEnabled(True)

    def bbox_setting(self):
        if self.pushButton_bboxSetting.text().endswith('Pausing'):
            self.progressBar.setFormat('RE-DRAW')
            self.pushButton_bboxSetting.setText("&B-box Reconfiguring")
            self.pushButton_algorithmProcessing.setEnabled(True)
            self.pushButton_locationLoading.setEnabled(False)
            self.pushButton_videoLoading.setEnabled(False)
            self.INIT = self.N
            self.rectList = []
            self.label_image.setCursor(QCursor(Qt.CrossCursor))
            self.isPainting = True
            self.paint_frame = self.get_frame(self.video_name, self.N)
            for id, bbox in self.info_store[self.INIT].items():
                x, y, w, h = bbox
                cv2.putText(self.paint_frame, str(id), (x, y-5), \
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
            return
        if self.isPainting:
            if max(self.info_store.keys(), default=-1) >= self.INIT:
                print('[INFO] Throw away the setting bbox at the frame of: ' + str(self.INIT))
                if self.INIT not in self.info_store.keys():
                    self.info_store[self.INIT] = {}
                if self.first_frame:
                    self.info_store[self.INIT] = {}
                    self.info['object_ids'] = []
                else:
                    def info_setting(id):
                        if id not in self.info['sequence_object_ids']:
                            self.info['sequence_object_ids'].append(id)
                        if id in self.info['init_bbox']:
                            self.info['init_bbox'].pop(id)
                        if id in self.info['init_object_ids']:
                            self.info['init_object_ids'].remove(id)
                            self.info['sequence_object_ids'].remove(id)
                    if self.spinBox.value() == 0:
                        for id in self.info_store[self.INIT].keys():
                            info_setting(id)
                        self.info_store[self.INIT] = {}
                    else:
                        try:
                            id = self.spinBox.value()
                            info_setting(id)
                            self.info_store[self.INIT].pop(id)
                        except:
                            print('[INFO] Throwing away this id is already done')
            self.rectList = []
            try:
                for id, bbox in self.info_store[self.INIT].items():
                    x, y, w, h = bbox
                    cv2.putText(self.paint_frame, str(id), (x, y-5), \
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                pass
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
        elif self.pushButton_bboxSetting.text().endswith('Setting'):
            self.info['object_ids'] = []
            self.progressBar.setFormat('DRAW')
            self.label_image.setCursor(QCursor(Qt.CrossCursor))
            if self.bbox_tips:
                reply = QMessageBox.information(self, 'Tips', 'You Can Set Initial Frame By: ' +
                                                'Keyboard \",\" - Previous Frame; ' +
                                                'Keyboard \".\" - Next Frame. ' +
                                                '(If You Need)', QMessageBox.Ignore, QMessageBox.Ok)
                if reply == QMessageBox.Ignore:
                    self.bbox_tips = False
            self.isPainting = True
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
            self.pushButton_bboxSetting.setText("&B-box Reconfiguring")
            self.pushButton_algorithmProcessing.setEnabled(True)
        else:
            raise Exception

    def algorithm_processing(self):
        if self.first_frame:
            # Set the info
            self.info = {}
            self.info['init_bbox'] = OrderedDict()
            self.info['init_object_ids'] = []
            if self.INIT not in self.info_store or len(self.info_store[self.INIT]) < 1:
                print('[INFO] Error in b-box choosing.')
                QMessageBox.information(self, 'Warning', \
                    'You Must Confirm Available B-boxes. (Shortcut Key: Alt + B)', QMessageBox.Ok)
                return
            for id, bbox in self.info_store[self.INIT].items():
                self.info['init_bbox'][id] = bbox
                self.info['init_object_ids'].append(id)
            self.info['object_ids'] = deepcopy(self.info['init_object_ids'])
            self.info['sequence_object_ids'] = []
            self.spinBox.setEnabled(True)
        
        if not self.check_bboxlist():
            print('[INFO] Error in b-box choosing.')
            QMessageBox.information(self, 'Warning', \
                'You Must Confirm Available B-boxes. (Shortcut Key: Alt + B)', QMessageBox.Ok)
            return
        
        self.pushButton_bboxSetting.setText("&B-box Pausing")
        if path.isfile(self.video_name):
            self.pushButton_videoLoading.setText('&Stream Suspending')
            self.pushButton_locationLoading.setEnabled(False)
            self.pushButton_videoLoading.setEnabled(True)
        elif path.exists(self.video_name):
            self.pushButton_locationLoading.setText('&Stream Suspending')
            self.pushButton_videoLoading.setEnabled(False)
            self.pushButton_locationLoading.setEnabled(True)
        else:
            raise Exception
        self.comboBox.setEnabled(False)
        self.pushButton_algorithmProcessing.setEnabled(False)
        self.label_image.setCursor(QCursor(Qt.ArrowCursor))
        self.isPainting = False
        self.progressBar.setFormat('INITIALIZE')
        print("[INFO] Starting pictures stream.")
        save_loc_d = './ans/' + self.video_name.split('/')[-1]
        save_loc_t = save_loc_d + '/' + 'aaa'#str(self.tracker_name)
        self.save_location = save_loc_t
        self.N = self.INIT

        for frame in self.get_frames(self.video_name):
            if self.first_frame:
                out = self.tracker.initialize(frame, self.info)
                if out is None:
                    out = {}
                self.prev_output = OrderedDict(out)
                self.info['init_object_ids'] = []
                self.first_frame = False
                # fps_cal = FPS().start()
                # self.analysis_init()
            else:
                if self.pushButton_locationLoading.text().endswith('Loading') and \
                    self.pushButton_videoLoading.text().endswith('Loading'):
                    # Suspending
                    self.comboBox.setEnabled(True)
                    self.spinBox.setEnabled(False)
                    self.selectBox.setEnabled(False)
                    self.scrollBar.setEnabled(False)
                    self.textBrowser.append('————————————\nSuspended Stream:\n' + self.video_name)
                    self.video_name = ''
                    break 
                if self.pushButton_bboxSetting.text().endswith('Reconfiguring'):
                    # Pausing
                    return 
                
                self.N += 1
                self.info_store[self.N] = {}
                
                self.info['previous_output'] = self.prev_output
                out = self.tracker.track(frame, self.info)
                if out is None:
                    out = {}
                self.prev_output = OrderedDict(out)
                self.info['init_object_ids'] = []
                self.info['sequence_object_ids'] = []
                
                if self.video_activate:
                    if 'segmentation' in out:
                        frame = overlay_mask(frame, out['segmentation'])
                    if 'target_polygon' in out:
                        for obj_id, state in out['target_polygon'].items():
                            state = [int(s) for s in state]
                            polygon = np.array(state).astype(np.int32)
                            cv2.polylines(frame, [polygon.reshape((-1, 1, 2))], \
                                True, (0, 255, 0), 2)
                            polygon_xmean = (polygon[0] + polygon[2] + polygon[4] + polygon[6]) / 4
                            polygon_ymean = (polygon[1] + polygon[3] + polygon[5] + polygon[7]) / 4
                            cv2.rectangle(frame, (int(polygon_xmean) - 1, int(polygon_ymean) - 1), \
                                (int(polygon_xmean) + 1, int(polygon_ymean) + 1), (0, 255, 0), 2)
                            cv2.putText(frame, str(obj_id), (polygon[6], polygon[7]), \
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    elif 'target_bbox' in out:
                        for obj_id, state in out['target_bbox'].items():
                            state = [int(s) for s in state]
                            cv2.rectangle(frame, (state[0], state[1]), \
                                (state[2] + state[0], state[3] + state[1]), \
                                (0, 255, 0), 2)
                            cv2.putText(frame, str(obj_id), (state[0], state[1]-5), \
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if 'object_presence_score' in out:
                        for obj_id, score in out['object_presence_score'].items():
                            state = out['target_bbox'][obj_id]
                            state = [int(s) for s in state]
                            cv2.putText(frame, '{:.2f}'.format(score), (state[0]+state[2]-20, \
                                state[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if 'time' in out:
                        if self.video_fps:
                            time_arr = np.array(list(out['time'].values()))
                            if self.W >= 200 and self.H >= 200:
                                cv2.putText(frame, 'FPS: {:.2f}'.format(1.0 / np.sum(time_arr)), \
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                            if self.W >= 400 and self.H >= 200:
                                cv2.putText(frame, 'FPSID: {:.2f}'.format(1.0 / np.mean(time_arr)), \
                                    (220, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    if 'target_bbox' in out:
                        for obj_id, state in out['target_bbox'].items():
                            bbox = list(map(int, state))
                            self.info_store[self.N][obj_id] = bbox
                    else:
                        Exception
                    frame = self.cv2_to_factor(frame, self.video_factor)
                    self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
                    QApplication.processEvents()

                # for b in range(len(self.bbox_list)):
                #     self.behavior_analysis(frame, b,
                #                            (polygon_xmean, polygon_ymean), (mask > 0))
                #     bbox = tuple(list(map(int, outputs['bbox'])))
                #     self.bbox_list_predict[self.N].append(bbox)

                # frame_draw = np.zeros(frame.shape, dtype=np.uint8)
                # last_xmean = []
                # last_ymean = []
                # last_num = 100
                # if len(self.bbox_list_predict) > last_num:
                #     c_bbox = self.bbox_list_predict[-last_num:]
                # else:
                #     c_bbox = self.bbox_list_predict
                # for c_num, bbox in enumerate(c_bbox):
                #     for b_num, b_bbox in enumerate(bbox):
                #         xmean = int(b_bbox[0] + (b_bbox[2] / 2))
                #         ymean = int(b_bbox[1] + (b_bbox[3] / 2))
                #         cv2.rectangle(frame_draw, (xmean - 1, ymean - 1), (xmean + 1, ymean + 1), (0, 192, 0), 2)
                #         if c_num > 0:
                #             cv2.line(frame_draw, (last_xmean[b_num], last_ymean[b_num]), (xmean, ymean), (0, 192, 0), 2)
                #             last_xmean[b_num] = xmean
                #             last_ymean[b_num] = ymean
                #         else:
                #             last_xmean.append(xmean)
                #             last_ymean.append(ymean)
                # frame = cv2.addWeighted(frame, 1, frame_draw, 0.5, 0)

                # fps_cal.update()
                # fps_cal.stop()
                # text = "FPS: {:.2f}".format(fps_cal.fps())
                # cv2.putText(frame, text, (60, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # if checkBoxcheckState:
                #     save_loc_i = save_loc_t + "/" + str(self.N).zfill(4) + ".jpg"
                #     cv2.imwrite(save_loc_i, frame)
                #     save_loc_m = save_loc_t + "/mask/" + str(self.N).zfill(4) + ".jpg"
                #     cv2.imwrite(save_loc_m, mask)
                
            
            self.progressBar.setProperty('value', (self.N - self.INIT) * 100 / (self.F - 1 - self.INIT))
            self.progressBar.setFormat('HANDLE: {:.2f}%'.format(100.0 * self.N / (self.F - 1)))
            self.scrollBar.setProperty('value', self.N)
            QApplication.processEvents()

        self.rectList = []
        print("[INFO] Ending pictures stream.")
        if checkBoxcheckState:
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
                self.progressBar.setFormat('PAUSING: %p%')
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
                if path.isfile(self.video_name):
                    cap = cv2.VideoCapture(self.video_name)
                    wri = cv2.VideoWriter(save_loc_t + '/video.avi', fourcc,
                                          int(cap.get(5)), (int(cap.get(3)), int(cap.get(4))))
                elif path.exists(self.video_name):
                    frame_0 = self.get_frame(self.video_name, 0)
                    wri = cv2.VideoWriter(save_loc_t + '/video.avi', fourcc,
                                          30, (frame_0.shape[1], frame.shape[0]))
                else:
                    raise Exception                    
                for j in range(last_j):
                    frame = cv2.imread(save_loc_t + '/' + str(j).zfill(4) + ".jpg")
                    wri.write(frame)
                    self.progressBar.setProperty('value', j * 100 / (last_j - 1))
                    self.progressBar.setFormat('SAVE: %p% [{:d}/{:d}]'.format(j, last_j - 1))
                    QApplication.processEvents()
                wri.release()
                self.pushButton_bboxSetting.setText('&B-box Setting')
                QApplication.processEvents()

        self.pushButton_locationLoading.setText('&Location Loading')
        self.pushButton_videoLoading.setText('&Video Loading')
        self.pushButton_locationLoading.setEnabled(True)
        self.pushButton_videoLoading.setEnabled(True)
        self.pushButton_bboxSetting.setEnabled(False)
        self.comboBox.setEnabled(True)
        self.progressBar.setFormat('STAND BY')
        self.progressBar.setProperty('value', 0)


    def combo_change(self):
        self.model_name = self.comboBox.currentText()
        # Ban combo
        self.comboBox.setEnabled(False)
        self.comboBoxThread = ComboBoxThread(self, self.progress_childSignal)
        self.comboBoxThread.start()

    def progress_child_change(self, num):
        if num == -1:
            self.comboBox.setEnabled(True)
            self.textBrowser.append('————————————\nError Model Name: ' + self.model_name)
        elif 0 <= num <= 100:
            self.progressBar.setFormat('HANDLE: {:d}%'.format(num))
            self.progressBar.setProperty('value', num)
        elif num == 101:
            # Activate buttons
            self.pushButton_locationLoading.setEnabled(True)
            self.pushButton_videoLoading.setEnabled(True)
            self.pushButton_outputConfiguring.setEnabled(True)
            self.comboBox.setEnabled(True)
            if self.video_name == '':
                self.progressBar.setFormat('STAND BY')
            else:
                self.progressBar.setFormat('STAND BY DATA')
            self.progressBar.setProperty('value', 0)
            self.textBrowser.append('————————————\nLoaded Model: ' + self.model_name)

    def spin_change(self):
        if not self.first_frame:
            max_id = max(self.info['object_ids'])
            self.spinBox.setMaximum(max_id)
    
    def wheelEvent(self, event):
        if self.ctrlPressed:
            if event.angleDelta().y() > 0:
                self.video_factor += 0.1
            elif event.angleDelta().y() < 0:
                self.video_factor -= 0.1
        else:
            return super().wheelEvent(event)

    def keyPressEvent(self, event):
        if self.isPainting:
            if event.key() == Qt.Key_Period:
                if self.scrollBar.value() < self.F - 1:
                    self.selectBox.setProperty('value', self.INIT + 1)
                    self.perm = True
            elif event.key() == Qt.Key_Comma:
                if self.scrollBar.value() > 0:
                    self.selectBox.setProperty('value', self.INIT - 1)
                    self.perm = True
        if event.key() == Qt.Key_Control:
            self.ctrlPressed = True
        else:
            return super().keyPressEvent(event)
        # elif (self.isStatus is 0) and checkBoxcheckState:
        #     if event.key() == Qt.Key_Period:
        #         if self.scrollBar.value() < self.F - 1:
        #             self.scrollBar.setProperty('value', self.scrollBar.value() + 1)
        #     elif event.key() == Qt.Key_Comma:
        #         if self.scrollBar.value() > self.INIT + 1:
        #             self.scrollBar.setProperty('value', self.scrollBar.value() - 1)
    
    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Control:
            self.ctrlPressed = False
        else:
            return super().keyReleaseEvent(event)

    def select_change(self):
        if self.isPainting:
            self.scrollBar.setProperty('value', self.selectBox.value())

    def slider_change(self):
        self.selectBox.setProperty('value', self.scrollBar.value())
        if self.isPainting:
            self.INIT = self.scrollBar.value()
            now_value = self.scrollBar.value()
            if getattr(self, 'scrollBarThread', False):
                if not self.scrollBarThread.isRunning():
                    self.scrollBarThread = ScrollBarThread(now_value, self)
                    self.scrollBarThread.start()
            else:
                self.scrollBarThread = ScrollBarThread(now_value, self)
                self.scrollBarThread.start()

    def slider_true_change(self):
        if self.isPainting:
            self.rectList = []
            if not self.first_frame and getattr(self.tracker, 'initialized_ids', False):
                for id in self.tracker.initialized_ids:
                    if id not in self.info['sequence_object_ids']:
                        self.info['sequence_object_ids'].append(id)
                    if self.INIT in self.info_store.keys() and \
                        id in self.info_store[self.INIT].keys():
                        self.info['init_bbox'][id] = self.info_store[self.INIT][id]
                    else:
                        if id in self.info['init_bbox']:
                            self.info['init_bbox'].pop(id)
            frame = self.get_frame(self.video_name, self.INIT)
            frame = self.cv2_to_factor(frame, self.video_factor)
            self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
            self.paint_frame = frame
            try:
                for id, bbox in self.info_store[self.INIT].items():
                    x, y, w, h = bbox
                    cv2.putText(self.paint_frame, str(id), (x, y-5), \
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            except:
                pass
            self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
            QApplication.processEvents()
        # elif (self.isStatus is 0) and checkBoxcheckState:
        #     if self.scrollBar.value() <= self.INIT:
        #         self.scrollBar.setProperty('value', self.INIT + 1)
        #     else:
        #         save_loc_i = self.save_location + '/' + str(self.scrollBar.value() - self.INIT).zfill(4) + ".jpg"
        #         frame = cv2.imread(save_loc_i)
        #         self.label_image.setPixmap(QPixmap(self.cv2_to_qimge(frame)))
        #         self.paint_frame = frame
        #         self.tempPix = QPixmap((self.cv2_to_qimge(self.paint_frame)))
        #         QApplication.processEvents()

    # def checkbox_change(self):
    #     if not self.pushButton_algorithmProcessing.isEnabled() or (self.isStatus is 3):
    #         if self.checkBox.checkState():
    #             self.checkBox.setChecked(False)
    #         else:
    #             self.checkBox.setChecked(True)
    #         QMessageBox.information(self, 'Warning', 'You Must Change Checkbox Before ' +
    #                                 '/ After the Algorithm Procedure.', QMessageBox.Ok)
    #         QApplication.processEvents()
    #     if self.checkBox.checkState():
    #         if self.save_tips:
    #             reply = QMessageBox.information(self, 'Tips', 'You Can View Saved Answer By: ' +
    #                                             'Keyboard \",\" - Previous Frame; ' +
    #                                             'Keyboard \".\" - Next Frame. ' +
    #                                             '(After Algorithm Procedure)', QMessageBox.Ignore, QMessageBox.Ok)
    #             if reply == QMessageBox.Ignore:
    #                 self.save_tips = False
