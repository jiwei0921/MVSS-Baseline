import numpy as np
import torch
import random

from PIL import Image, ImageOps
from torchvision.transforms import Resize, ToTensor, InterpolationMode


class Relabel:

    def __init__(self, olabel, nlabel):
        self.olabel = olabel
        self.nlabel = nlabel

    def __call__(self, tensor):
        assert (isinstance(tensor, torch.LongTensor) or isinstance(tensor, torch.ByteTensor)) , 'tensor needs to be LongTensor'
        tensor[tensor == self.olabel] = self.nlabel
        return tensor


class ToLabel:

    def __call__(self, image):
        return torch.from_numpy(np.array(image)).long().unsqueeze(0)


class Colorize:

    def __init__(self, colormap, num_classes):
        self.cmap = colormap
        self.cmap[num_classes] = self.cmap[-1]
        self.cmap = torch.from_numpy(self.cmap[:num_classes])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = gray_image[0] == label
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


# Apply same transformations on all inputs and labels
class MyCoTransform(object):

    def __init__(self, target_transform, shallow_dec, augment, work_res, random_crop):
        self.target_transform = target_transform
        self.shallow_dec = shallow_dec
        self.augment = augment
        self.work_res = work_res
        self.random_crop = random_crop
        pass

    def __call__(self, inputs_list, thermals_list, labels_list):
        # do something to both images

        if self.augment:
            # Random hflip
            h_flip = random.random()
            if h_flip < 0.5:
                for i, val in enumerate(inputs_list):
                    inputs_list[i] = val.transpose(Image.FLIP_LEFT_RIGHT)
                    thermals_list[i] = thermals_list[i].transpose(Image.FLIP_LEFT_RIGHT)
                    if labels_list[i] is not None:
                        labels_list[i] = labels_list[i].transpose(Image.FLIP_LEFT_RIGHT)

            # Random translation 0-2 pixels (fill rest with padding
            trans_x = random.randint(-2, 2)
            trans_y = random.randint(-2, 2)

            for i, val in enumerate(inputs_list):
                inputs_list[i] = ImageOps.expand(val, border=(trans_x, trans_y, 0, 0), fill=0)
                thermals_list[i] = ImageOps.expand(thermals_list[i], border=(trans_x, trans_y, 0, 0), fill=0)
                # pad label filling with 255
                if labels_list[i] is not None:
                    labels_list[i] = ImageOps.expand(labels_list[i], border=(trans_x, trans_y, 0, 0), fill=255)
            for i, val in enumerate(inputs_list):
                crop_tuple = (0, 0, val.size[0] - trans_x, val.size[1] - trans_y)
                inputs_list[i] = val.crop(crop_tuple)
                thermals_list[i] = thermals_list[i].crop(crop_tuple)
                if labels_list[i] is not None:
                    assert(labels_list[i].size[0] == val.size[0] and labels_list[i].size[1] == val.size[1])
                    labels_list[i] = labels_list[i].crop(crop_tuple)

        # PSP training with smaller crops
        if self.augment and self.random_crop:

            if_crop = random.random()
            if if_crop < 0.5:
                randcrop_x = random.randint(0, inputs_list[0].size[0] - inputs_list[0].size[0]*0.8)
                randcrop_y = random.randint(0, inputs_list[0].size[1] - inputs_list[0].size[1]*0.8)
                crop_tuple = (randcrop_x, randcrop_y,
                              randcrop_x + inputs_list[0].size[0]*0.8, randcrop_y + inputs_list[0].size[1]*0.8)

                for i, val in enumerate(inputs_list):
                    inputs_list[i] = val.crop(crop_tuple)
                    thermals_list[i] = thermals_list[i].crop(crop_tuple)
                    inputs_list[i] = Resize(self.work_res, InterpolationMode.BILINEAR)(inputs_list[i])
                    thermals_list[i] = Resize(self.work_res, InterpolationMode.BILINEAR)(thermals_list[i])
                    if labels_list[i] is not None:
                        assert(labels_list[i].size[0] == val.size[0] and labels_list[i].size[1] == val.size[1])
                        labels_list[i] = labels_list[i].crop(crop_tuple)
                        labels_list[i] = Resize(self.work_res, InterpolationMode.NEAREST)(labels_list[i])

        for i, val in enumerate(inputs_list):
            inputs_list[i] = ToTensor()(val)
            thermals_list[i] = ToTensor()(thermals_list[i])

        if self.shallow_dec:
            for i, val in enumerate(labels_list):
                if labels_list[i] is not None:
                    labels_list[i] = Resize((int(self.work_res[0] / 8),
                                             int(self.work_res[1] / 8)), InterpolationMode.NEAREST)(val)

        return inputs_list,thermals_list,labels_list
