from ..utils import TrackerParams
import os
from ...config.stark_lightning_X_trt.config import cfg, update_config_from_file


root_abspath = os.path.abspath(__file__).split('lib')[0]

def parameters(yaml_name: str):
    params = TrackerParams()
    prj_dir = root_abspath
    save_dir = root_abspath
    # update default config from yaml file
    yaml_file = os.path.join(prj_dir, 'experiments/stark_lightning_X_trt/%s.yaml' % yaml_name)
    update_config_from_file(yaml_file)
    params.cfg = cfg
    print("test config: ", cfg)

    # template and search region
    params.template_factor = cfg.TEST.TEMPLATE_FACTOR
    params.template_size = cfg.TEST.TEMPLATE_SIZE
    params.search_factor = cfg.TEST.SEARCH_FACTOR
    params.search_size = cfg.TEST.SEARCH_SIZE

    # Network checkpoint path
    params.checkpoint = os.path.join(save_dir,
                                     "checkpoints/train/stark_lightning_X_trt/%s/STARKLightningXtrt_ep%04d.pth.tar" %
                                     (yaml_name, cfg.TEST.EPOCH))
    # whether to save boxes from all queries
    params.save_all_boxes = False
    return params
