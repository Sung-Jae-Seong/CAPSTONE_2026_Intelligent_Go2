import os
import sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEBLURGAN_DIR = os.path.normpath(os.path.join(ROOT_DIR, "DeblurGANv2"))

if not os.path.isdir(DEBLURGAN_DIR):
    DEBLURGAN_DIR = os.path.normpath(os.path.join(ROOT_DIR, "okiamtrash", "DeblurGANv2"))

if DEBLURGAN_DIR not in sys.path:
    sys.path.insert(0, DEBLURGAN_DIR)

from run_mobilenet_deblur import DEFAULT_WEIGHTS, MobileNetDeblurPredictor


class DeblurGANV2Deblurrer:
    def __init__(self, weights_path, device):
        self.predictor = MobileNetDeblurPredictor(
            weights_path=weights_path,
            device=device,
        )

    def deblur(self, rgb_image):
        return self.predictor.predict_rgb(rgb_image)


def build_deblurrer():
    weights_path = os.path.abspath(os.getenv("DEBLURGANV2_WEIGHTS", DEFAULT_WEIGHTS))
    device = os.getenv("DEBLURGANV2_DEVICE", "cuda")
    return DeblurGANV2Deblurrer(
        weights_path=weights_path,
        device=device,
    )
