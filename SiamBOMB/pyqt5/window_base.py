# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 

from os import path
from copy import deepcopy
from glob import glob
import cv2

from PyQt5.QtGui import QImage

from .window_ui import Ui_MainWindow


class BaseWindow(Ui_MainWindow):
    def __init__(self, parent=None):
        super(BaseWindow, self).__init__(parent)
        self.isDracula = False
        self.tracker = None

    def cv2_to_qimge(self, cvImg):
        cvImg = cv2.cvtColor(cvImg, cv2.COLOR_BGR2RGB)
        height, width, channel = cvImg.shape
        bytesPerLine = 3 * width
        qImg = QImage(cvImg.data, width, height, bytesPerLine, QImage.Format_RGB888)
        return qImg
    
    def cv2_to_factor(self, frame, factor):
        if not (0.99 < factor < 1.01):
            if self.H * factor > self.size().width():
                factor = 1.0 * self.size().width() / self.H
            frame = cv2.resize(frame, (0, 0), fx=factor, fy=factor)
        return frame

    def check_bboxlist(self, min_bbox_width=5):
        if len(self.info['object_ids']) < 1:
            return False
        elif getattr(self.tracker, 'initialized_ids', False) and len(self.tracker.initialized_ids) \
            - len(self.info['sequence_object_ids']) + len(self.info['init_bbox']) < 1:
            return False
        else:
            for bbox in self.info['init_bbox'].values():
                _, _, w, h = bbox
                if w < min_bbox_width or h < min_bbox_width:
                    return False
            return True
    
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
                            key=lambda x: int(x.split('/')[-1].split('.')[0]))
            if 0 <= frame_num < len(images):
                frame = cv2.imread(images[frame_num])
                return frame
            else:
                return