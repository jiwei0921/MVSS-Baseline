import sys
sys.path.append("..")

import cv2
import math
import numpy as np
import tqdm

from argparse import ArgumentParser
from torch.utils.data import DataLoader
from PIL import Image
from datasets.helpers import DATASETS_DICT


def main(args):

    subsets = ["train"]
    histograms = []
    total_pixels = 0
    for subset in subsets:

        dataset = DATASETS_DICT[args.dataset](args, subset, co_transform=False, interval=[0, 0])
        loader = DataLoader(dataset, num_workers=24, batch_size=1, shuffle=False)
        hist = np.zeros([256, 1])
        for step, (images, thermals, labels, orig_images, orig_thermals, orig_labels,
               file_path, file_path_gt,
               images_filenames, labels_filenames) in enumerate(tqdm.tqdm(loader)):

            img = Image.open(file_path_gt[0]).convert('P')
            img = np.array(img)
            hist += cv2.calcHist([img], [0], None, [256], [0, 256])
            total_pixels += img.shape[0] * img.shape[1]

        histograms.append(hist[0:dataset.num_classes])

    # TRAIN (+ VAL)
    total_hist = histograms[0] # + histograms[1]
    # print("total_hist\n", total_hist)
    normed_hist = total_hist / total_pixels
    # print("normed_hist\n", normed_hist)
    # print("normed_hist SUM:", np.sum(normed_hist))
    norm_value = 1.10 if args.train_decoder else 1.20

    for i, elt in enumerate(normed_hist):
        normed_hist[i] = 1.0 / math.log(norm_value + normed_hist[i])

    print("weights hist\n", normed_hist)


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--dataset', default='MVSeg')
    parser.add_argument('--backbone', type=str, default="deeplab50")
    parser.add_argument('--use-orig-res', action='store_true')
    parser.add_argument('--train-decoder', action='store_true')
    parser.add_argument('--random-crop', action='store_true')

    main(parser.parse_args())



