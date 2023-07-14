import os
import numpy as np

from torchvision.transforms import Compose, Normalize

from datasets.generic import GenericDataset
from datasets.transform import Relabel, ToLabel

###############################################################################################
# Sets the dataset path

MVSeg_ROOT = "/set_your_path/MVSeg_Dataset/"

###############################################################################################

normalize_tensor_mvseg = Compose([Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

target_transform_mvseg = Compose([ToLabel(), Relabel(255, 25)])

MVSeg_CLASSES_DICT = {
    "Background"       : {"weight_enc": 1.0, "weight": 39.31534571},   # 0
    "Car"              : {"weight_enc": 1.0, "weight": 12.26980802},   # 1
    "Bus"              : {"weight_enc": 1.0, "weight": 42.18054118},   # 2
    "Motorcycle"       : {"weight_enc": 1.0, "weight": 46.44039747},   # 3
    "Bicycle"          : {"weight_enc": 1.0, "weight": 42.15347271},   # 4
    "Pedestrian"       : {"weight_enc": 1.0, "weight": 37.82259361},   # 5
    "Motorcyclist"     : {"weight_enc": 1.0, "weight": 48.68300575},   # 6
    "Bicyclist"        : {"weight_enc": 1.0, "weight": 48.51023962},   # 7
    "Cart"             : {"weight_enc": 1.0, "weight": 48.83860785},   # 8
    "Bench"            : {"weight_enc": 1.0, "weight": 49.28115286},   # 9
    "Umbrella"         : {"weight_enc": 1.0, "weight": 48.6147525 },   # 10
    "Box"              : {"weight_enc": 1.0, "weight": 46.51266704},   # 11
    "Pole"             : {"weight_enc": 1.0, "weight": 36.7427391 },   # 12
    "Street_lamp"      : {"weight_enc": 1.0, "weight": 45.31953854},   # 13
    "Traffic_light"    : {"weight_enc": 1.0, "weight": 47.76498565},   # 14
    "Traffic_sign"     : {"weight_enc": 1.0, "weight": 40.85773736},   # 15
    "Car_stop"         : {"weight_enc": 1.0, "weight": 45.25310069},   # 16
    "Color_cone"       : {"weight_enc": 1.0, "weight": 47.60953912},   # 17
    "Sky"              : {"weight_enc": 1.0, "weight": 9.86157348 },   # 18
    "Road"             : {"weight_enc": 1.0, "weight": 4.97322828 },   # 19
    "Sidewalk"         : {"weight_enc": 1.0, "weight": 24.06969199},   # 20
    "Curb"             : {"weight_enc": 1.0, "weight": 32.05760907},   # 21
    "Vegetation"       : {"weight_enc": 1.0, "weight": 4.88157226 },   # 22
    "Terrain"          : {"weight_enc": 1.0, "weight": 8.83349187 },   # 23
    "Building"         : {"weight_enc": 1.0, "weight": 7.78494492 },   # 24
    "Ground"           : {"weight_enc": 1.0, "weight": 6.72750005 }    # 25
}


def colormap_mvseg():
    cmap_mat = np.zeros([256, 3]).astype(np.uint8)
    a_map = cmap()
    for i in range(26):
        cmap_mat[i,:] = np.array(a_map[i])
    return cmap_mat


def cmap():
    return [
        (0, 0, 0),          # 0:    background(unlabeled)
        (0, 0, 142),        # 1:    Car
        (0, 60, 100),       # 2:    Bus
        (0, 0, 230),        # 3:    Motorcycle
        (119, 11, 32),      # 4:    Bicycle
        (255, 0, 0),        # 5:    Pedestrian
        (0, 139, 139),      # 6:    Motorcyclist
        (255, 165, 150),    # 7:    Bicyclist
        (192, 64, 0),       # 8:    Cart
        (211, 211, 211),    # 9:    Bench
        (100, 33, 128),     # 10:   Umbrella
        (117, 79, 86),      # 11:   Box
        (153, 153, 153),    # 12:   Pole
        (190, 122, 222),    # 13:   Street_lamp
        (250, 170, 30),     # 14:   Traffic_light
        (220, 220, 0),      # 15:   Traffic_sign
        (222, 142, 35),     # 16:   Car_stop
        (205, 155, 155),    # 17:   Color_cone
        (70, 130, 180),     # 18:   Sky
        (128, 64, 128),     # 19:   Road
        (244, 35, 232),     # 20:   Sidewalk
        (0, 0, 70),         # 21:   Curb
        (107, 142, 35),     # 22:   Vegetation
        (152, 251, 152),    # 23:   Terrain
        (70, 70, 70),       # 24:   Building
        (110, 80, 100)      # 25:   Ground
        ]


class MVSeg(GenericDataset):

    def __init__(self, args, subset, co_transform, shallow_dec=False, augment=False, interval=None, load_train_ids=True, print_all_logs=True):

        with open(os.path.join(MVSeg_ROOT, subset + '.txt')) as f:
            lines = f.readlines()
            videolists = [line.strip() for line in lines]

        filenames_gt = []
        filenames = []
        filenames_ir = []   # ir means infrared
        for video in videolists:
            # Create List for only labeled GT
            label_path = os.path.join(MVSeg_ROOT, 'data', video, 'label')
            filenames_gt_i = [os.path.join(label_path, f) for f in os.listdir(label_path)
                              if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            filenames_gt += sorted(filenames_gt_i)

            # Create List for only labeled Image seqs
            image_path = os.path.join(MVSeg_ROOT, 'data', video, 'visible')
            file_postfix = os.listdir(image_path)[-1][-4:]
            filenames_i = [os.path.join(image_path, f[:-5] + 'v' + file_postfix) for f in os.listdir(label_path)
                           if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            filenames += sorted(filenames_i)

            # Create List for only labeled Thermal seqs
            thermal_path = os.path.join(MVSeg_ROOT, 'data', video, 'infrared')
            file_ir_postfix = os.listdir(thermal_path)[-1][-4:]
            filenames_ir_i = [os.path.join(thermal_path, f[:-5] + 'i' + file_ir_postfix) for f in os.listdir(label_path)
                           if any(f.endswith(ext) for ext in ['.jpg', '.png'])]
            filenames_ir += sorted(filenames_ir_i)


        classes_dict = MVSeg_CLASSES_DICT

        orig_res = (320, 480)
        if args.backbone == "deeplab50":
            work_res = (320, 480) #(480, 640)
        else:
            work_res = (-1, -1)
            print("unknown backbone")
            exit(0)

        target_transform = target_transform_mvseg      # Convert it to long-type tensor
        normalize_tensor = normalize_tensor_mvseg

        super(MVSeg, self).__init__(
            args, filenames, filenames_gt, filenames_ir, classes_dict,
            orig_res, work_res, target_transform, normalize_tensor, colormap_mvseg(),
            co_transform, shallow_dec, augment, interval, print_all_logs)

