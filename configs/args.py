import argparse
import configs.config as cfg
from math import isfinite, inf
from utils.sys_colors import BLUE, RESET


def strWithNone(v):
    return None if v == "None" else v


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def str2int_inf(v):
    value = float(v)

    if isfinite(value):
        return int(value)
    else:
        return inf


__argparser = argparse.ArgumentParser(prog="Voice Analysis")
__argparser.add_argument("--epochs", type=str2int_inf,
                         required=False, default=cfg.TRAINING.EPOCHS)
__argparser.add_argument("--reduced_dataset", type=str2int_inf,
                         required=False, default=None)
__argparser.add_argument("--load_optimizer", type=bool,
                         required=False, default=True)
__argparser.add_argument("--checkpoint", type=strWithNone,
                         required=False, default=None)
__argparser.add_argument("--checkpoint_type", type=str,
                         required=False, default="model")
__argparser.add_argument("--learning_rate", type=float,
                         required=False, default=cfg.TRAINING.LEARNING_RATE_INITIAL)
__argparser.add_argument("--batch_size", type=int,
                         required=False, default=cfg.TRAINING.BATCH_SIZE)
__argparser.add_argument("--testing_set", type=str2int_inf,
                         required=False, default=0)
__argparser.add_argument("--workers", type=int,
                         required=False, default=0)
__argparser.add_argument("--mixed_precision", type=str2bool,
                         required=False, default=True)

args = __argparser.parse_args()

print(BLUE + "---------------------------------------")
print("Launch Arguments:")
for arg in vars(args):
    value = str(getattr(args, arg))
    print("\t{:<20} {:>10}".format(arg, value))
print("---------------------------------------" + RESET)
