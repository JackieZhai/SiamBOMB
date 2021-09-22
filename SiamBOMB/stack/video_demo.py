import os
import sys
import argparse

prj_path = os.path.join(os.path.dirname(__file__), '..')
if prj_path not in sys.path:
    sys.path.append(prj_path)

from .lib.test.evaluation import Tracker


def run_video(tracker_name, tracker_param, videofile, optional_box=None, debug=None, save_results=False):
    """Run the tracker on your webcam.
    args:
        tracker_name: Name of tracking method.
        tracker_param: Name of parameter file.
        debug: Debug level.
    """
    tracker = Tracker(tracker_name, tracker_param, "video")
    tracker.run_video(videofilepath=videofile, optional_box=optional_box, debug=debug, save_results=save_results)


root_dir = os.path.dirname(os.path.abspath(__file__))

def main():
    parser = argparse.ArgumentParser(description='Run the tracker on your webcam.')
    # stark_s, stark_st, stark_lightning_X_trt
    parser.add_argument('--tracker_name', default='stark_lightning_X_trt', type=str, help='Name of tracking method.')
    # baseline, baseline_got10k_only
    # baseline, baseline_R101, baseline_got10k_only, baseline_R101_got10k_only
    # baseline_rephead_4_lite_search5 (use_onnx=True, num_gpu=1)
    parser.add_argument('--tracker_param', default='baseline_rephead_4_lite_search5', type=str, help='Name of parameter file.')
    parser.add_argument('--videofile', default=os.path.join(root_dir, 'dataset/fishes_Test.mp4'), type=str, help='path to a video file.')
    parser.add_argument('--optional_box', default=None, type=float, nargs="+", help='optional_box with format x y w h.')
    parser.add_argument('--debug', default=0, type=int, help='Debug level.')
    parser.add_argument('--save_results', dest='save_results', action='store_true', help='Save bounding boxes')
    parser.set_defaults(save_results=False)

    args = parser.parse_args()

    run_video(args.tracker_name, args.tracker_param, args.videofile, args.optional_box, args.debug, args.save_results)


if __name__ == '__main__':
    main()
