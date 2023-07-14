import torch

from datasets.mvss_dataset import MVSeg, MVSeg_CLASSES_DICT


DATASETS_DICT = {
    "MVSeg": MVSeg,
}

DATASETS_NUM_CLASSES = {
    "MVSeg": len(MVSeg_CLASSES_DICT),
}

DATASETS_CLASSES_DICT = {
    "MVSeg": MVSeg_CLASSES_DICT,
}


def classes_weights(classes_dict, shallow_dec, gpu):

    weights = torch.ones(len(classes_dict))

    for i, key in enumerate(classes_dict):
        weights[i] = classes_dict[key]["weight_enc"] if shallow_dec else classes_dict[key]["weight"]
    return weights.cuda(gpu)

