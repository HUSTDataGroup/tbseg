from .TBSeg import build as build_TBSeg


def build_model(args):
    if args.model == "TBSeg":
        return build_TBSeg(args)
    else:
        raise ValueError("invalid model:{}".format(args.model))