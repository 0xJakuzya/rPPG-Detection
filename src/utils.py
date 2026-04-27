from models import CHROM, POS
from src import config

def make_algorithm(fps: float):
    if config.RPPG_METHOD.upper() == "CHROM":
        return CHROM(fps)
    return POS(fps)
