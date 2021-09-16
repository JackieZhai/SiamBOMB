# 
# GitHub: JackieZhai @ MiRA, CASIA, Beijing, China
# See Corresponding LICENSE, All Right Reserved.
# 

import sys
from os import path
import gdown


url_siamme = 'https://drive.google.com/u/0/uc?id=1VVpCAUJeysyRWdLdfW1IsT3AsQUQvwAU'
url_kys = 'https://drive.google.com/u/0/uc?id=13uOa9cpTyVf7hB3RkdjjN-hyEr4yJiSw'
url_lwl = 'https://drive.google.com/u/0/uc?id=18G1kAcLrTgO1Hgyj290mqPKsfVIkI9lw'
url_keept = 'https://drive.google.com/u/0/uc?id=1zyadmon8codJDvOQlHAsBDPA_ouN4Zud'
url_sdimps = 'https://drive.google.com/u/0/uc?id=1cRgzZ0MFFeE2PaZL3BWbYXu9Aq73f-TR'

root_abs = path.dirname(path.abspath(__file__))
path_siamme = path.join(root_abs, 'pysot/experiments/siammaske_r50_l3', 'model.pkl')
path_kys = path.join(root_abs, 'pytracking/networks', 'kys.pth')
path_lwl = path.join(root_abs, 'pytracking/networks', 'lwl_boxinit.pth')
path_keept = path.join(root_abs, 'pytracking/networks', 'keep_track.pth')
path_sdimps = path.join(root_abs, 'pytracking/networks', 'super_dimp_simple.pth')


if __name__ == '__main__':
    gdown.download(url_siamme, path_siamme, quiet=False)
    gdown.download(url_kys, path_kys, quiet=False)
    gdown.download(url_lwl, path_lwl, quiet=False)
    gdown.download(url_keept, path_keept, quiet=False)
    gdown.download(url_sdimps, path_sdimps, quiet=False)
