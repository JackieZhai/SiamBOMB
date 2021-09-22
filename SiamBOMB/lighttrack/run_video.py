import os
import cv2
import torch
import torch.utils.data
import random
import argparse
import numpy as np

import lib.models.models as models

from os.path import exists, join, dirname, realpath
from easydict import EasyDict as edict
from lib.utils.utils import load_pretrain, cxy_wh_2_rect, get_axis_aligned_bbox, load_dataset, poly_iou

from lib.tracker.lighttrack import Lighttrack


def parse_args():
    parser = argparse.ArgumentParser(description='Test LightTrack')
    parser.add_argument('--arch', default='LightTrackM_Subnet', dest='arch', help='backbone architecture')
    parser.add_argument('--resume', default='snapshot/LightTrackM/LightTrackM.pth', type=str, help='pretrained model')
    parser.add_argument('--video', default='dataset/fishes_Test.mp4', type=str, help='video file path')
    parser.add_argument('--stride', default=16, type=int, help='network stride')
    parser.add_argument('--even', default=0, type=int)
    parser.add_argument('--path_name', default='back_04502514044521042540+cls_211000022+reg_100000111_ops_32', type=str)
    parser.add_argument('--save', default=True, type=bool, help='save pictures')
    parser.add_argument('--init_bbox', default=None, help='bbox in the first frame None or [lx, ly, w, h]')
    args = parser.parse_args()

    return args


DATALOADER_NUM_WORKER = 2


class ImageDataset:
    def __init__(self, image_files):
        self.image_files = image_files

    def __getitem__(self, i):
        fname = self.image_files[i]
        im = cv2.imread(fname)
        if len(im.shape) == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training
        rgb_im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        return im, rgb_im

    def __len__(self):
        return len(self.image_files)


def collate_fn(x):
    return x[0]


def track(siam_tracker, siam_net, video, args):
    start_frame, toc = 0, 0
    snapshot_dir = os.path.dirname(args.resume)
    result_dir = os.path.join(snapshot_dir, '../..', 'result')
    model_name = snapshot_dir.split('/')[-1]

    # save result to evaluate
    tracker_path = os.path.join(result_dir, args.dataset, model_name)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    if not os.path.exists(tracker_path):
        os.makedirs(tracker_path)

    if 'VOT' in args.dataset:
        baseline_path = os.path.join(tracker_path, 'baseline')
        video_path = os.path.join(baseline_path, video['name'])
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        result_path = os.path.join(video_path, video['name'] + '_001.txt')
    else:
        result_path = os.path.join(tracker_path, '{:s}.txt'.format(video['name']))

    if os.path.exists(result_path):
        return

    regions = []
    lost = 0

    image_files, gt = video['image_files'], video['gt']

    dataset = ImageDataset(image_files)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn,
                                             num_workers=DATALOADER_NUM_WORKER)

    with torch.no_grad():
        for f, x in enumerate(dataloader):
            im, rgb_im = x
            if len(im.shape) == 2:
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)  # align with training

            tic = cv2.getTickCount()
            if f == start_frame:  # init
                cx, cy, w, h = get_axis_aligned_bbox(gt[f])

                target_pos = np.array([cx, cy])
                target_sz = np.array([w, h])

                state = siam_tracker.init(rgb_im, target_pos, target_sz, siam_net)  # init tracker

                regions.append(1 if 'VOT' in args.dataset else gt[f])
            elif f > start_frame:  # tracking
                state = siam_tracker.track(state, rgb_im)

                location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

                b_overlap = poly_iou(gt[f], location) if 'VOT' in args.dataset else 1
                if b_overlap > 0:
                    regions.append(location)
                else:
                    regions.append(2)
                    start_frame = f + 5
                    lost += 1
            else:
                regions.append(0)

            toc += cv2.getTickCount() - tic

    with open(result_path, "w") as fin:
        if 'VOT' in args.dataset:
            for x in regions:
                if isinstance(x, int):
                    fin.write("{:d}\n".format(x))
                else:
                    p_bbox = x.copy()
                    fin.write(','.join([str(i) for i in p_bbox]) + '\n')

    toc /= cv2.getTickFrequency()
    print('Video: {:12s} Time: {:2.1f}s Speed: {:3.1f}fps  Lost {}'.format(video['name'], toc, f / toc, lost))


def track_video(siam_tracker, siam_net, video_path, init_box=None, args=None):

    assert os.path.isfile(video_path), "please provide a valid video file"

    video_name = video_path.split('/')[-1]
    video_name = video_name.split('.')[0]
    save_path = os.path.join('vis', video_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cap = cv2.VideoCapture(video_path)
    display_name = 'Video: {}'.format(video_path.split('/')[-1])
    cv2.namedWindow(display_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
    cv2.resizeWindow(display_name, 960, 720)
    success, frame = cap.read()
    cv2.imshow(display_name, frame)

    if success is not True:
        print("Read failed.")
        exit(-1)

    # init
    count = 0

    if init_box is not None:
        lx, ly, w, h = init_box
        target_pos = np.array([lx + w/2, ly + h/2])
        target_sz = np.array([w, h])

        state = siam_tracker.init(frame, target_pos, target_sz, siam_net)  # init tracker

    else:
        while True:

            frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1, (0, 0, 255), 1)

            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])

            state = siam_tracker.init(frame_disp, target_pos, target_sz, siam_net)  # init tracker

            break

    while True:
        ret, frame = cap.read()

        if frame is None:
            return

        frame_disp = frame.copy()

        # Draw box
        state = siam_tracker.track(state, frame_disp)

        location = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        x1, y1, x2, y2 = int(location[0]), int(location[1]), int(location[0] + location[2]), int(location[1] + location[3])

        cv2.rectangle(frame_disp, (x1, y1), (x2, y2), (0, 255, 0), 5)

        font_color = (0, 0, 0)
        cv2.putText(frame_disp, 'Tracking!', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)
        cv2.putText(frame_disp, 'Press r to reset', (20, 55), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)
        cv2.putText(frame_disp, 'Press q to quit', (20, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1,
                   font_color, 1)

        # Display the resulting frame
        cv2.imshow(display_name, frame_disp)

        if args.save:
            save_name = os.path.join(save_path, '{:04d}.jpg'.format(count))
            cv2.imwrite(save_name, frame_disp)
            count += 1

        key = cv2.waitKey(1)
        # key = None
        if key == ord('q'):
            break
        elif key == ord('r'):
            ret, frame = cap.read()
            frame_disp = frame.copy()

            cv2.putText(frame_disp, 'Select target ROI and press ENTER', (20, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                       1.5,
                       (0, 0, 0), 1)

            cv2.imshow(display_name, frame_disp)
            lx, ly, w, h = cv2.selectROI(display_name, frame_disp, fromCenter=False)
            target_pos = np.array([lx + w / 2, ly + h / 2])
            target_sz = np.array([w, h])

            state = siam_tracker.init(frame_disp, target_pos, target_sz, siam_net)  # init tracker

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    info = edict()
    info.arch = args.arch
    info.dataset = 'VOT2019'
    info.epoch_test = False
    info.stride = args.stride

    siam_info = edict()
    siam_info.arch = args.arch
    siam_info.dataset = 'VOT2019'
    siam_info.epoch_test = False
    siam_info.stride = args.stride
    # build tracker
    siam_tracker = Lighttrack(siam_info, even=args.even)
    # build siamese network
    if args.path_name != 'NULL':
        siam_net = models.__dict__[args.arch](args.path_name, stride=siam_info.stride)
    else:
        siam_net = models.__dict__[args.arch](stride=siam_info.stride)

    siam_net = load_pretrain(siam_net, args.resume)
    siam_net.eval()
    siam_net = siam_net.cuda()

    track_video(siam_tracker, siam_net, args.video, init_box=args.init_bbox, args=args)


if __name__ == '__main__':
    main()
