# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from ..core.config import cfg
from ..tracker.siamrpn_tracker import SiamRPNTracker
from ..tracker.siammask_tracker import SiamMaskTracker
from ..tracker.siamrpnlt_tracker import SiamRPNLTTracker
from ..tracker.siammask_e_tracker import SiamMaskETracker

TRACKS = {
          'SiamRPNTracker': SiamRPNTracker,
          'SiamMaskTracker': SiamMaskTracker,
          'SiamRPNLTTracker': SiamRPNLTTracker,
          'SiamMaskETracker': SiamMaskETracker
         }


def build_tracker(model):
    return TRACKS[cfg.TRACK.TYPE](model)

