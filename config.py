import os
from yacs.config import CfgNode as CN

_C = CN()

# -----------------------------------------------------------------------------
# Common settings
# -----------------------------------------------------------------------------
_C.COMM = CN()

config = _C.clone()
config.defrost()

config.COMM.EXP_NAME = os.path.basename(os.getcwd())
config.COMM.EXP_TRAIN_LOG = os.path.join("./inference_log", config.COMM.EXP_NAME)
if not os.path.exists(config.COMM.EXP_TRAIN_LOG):
    print("Init Training LogFile...")
    os.makedirs(config.COMM.EXP_TRAIN_LOG)

config.freeze()


if __name__ == '__main__':
    import pprint
    pprint.pprint(config)
