import os
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Resize, InterpolationMode

from datasets.transform import MyCoTransform, Colorize
import utils.visualize


def root_check(path):
    if not os.path.exists(path):
        print("ERROR: Path does not exist: {}".format(path))
        print("Please make sure that the root path is correctly set for your dataset. "
              "For instance, MVSeg_ROOT is set in datasets/mvss_dataset.py")
        return False

    return True


class GenericDataset(Dataset):

    def __init__(self, args, filenames, filenames_gt, filenames_ir, classes_dict,
                 orig_res, work_res, target_transform, normalize_tensor, colormap,
                 co_transform, shallow_dec=False, augment=False, interval=None, print_all_logs=True):

        if interval is None:
            interval = [0, 0]
        assert(len(interval) == 2 and interval[0] >= 0 and interval[1] >= 0)

        self.baseline_mode = args.baseline_mode
        self.filenames = filenames
        self.filenames_gt = filenames_gt
        self.filenames_ir = filenames_ir
        self.classes_dict = classes_dict
        self.orig_res = orig_res
        self.work_res = work_res
        self.target_transform = target_transform
        self.normalize_tensor = normalize_tensor

        self.num_classes = len(classes_dict)
        self.colorize = Colorize(colormap, self.num_classes)

        self.backbone = args.backbone
        self.augment = augment
        self.random_crop = args.random_crop
        self.use_orig_res = args.use_orig_res
        self.interval = interval
        self.sample_rate = args.sample_rate

        self.filenames.sort()
        self.filenames_gt.sort()
        self.filenames_ir.sort()

        self.input_images_size = len(self.filenames)
        self.gt_images_size = len(self.filenames_gt)
        self.ir_images_size = len(self.filenames_ir)

        self.co_transform = None
        if co_transform:
            self.co_transform = MyCoTransform(self.target_transform, shallow_dec, augment=self.augment,
                                              work_res=self.work_res, random_crop=self.random_crop)
        if print_all_logs:
            utils.visualize.print_summary(self.__dict__, self.__class__.__name__)

    def __getitem__(self, index):
        file_path = self.filenames[index]
        file_path_gt = self.filenames_gt[index]
        file_path_ir = self.filenames_ir[index]

        labels = []
        images = []
        thermals=[]

        orig_labels = []
        orig_images = []
        orig_thermals=[]

        images_filenames = []
        labels_filenames = []
        thermals_filenames=[]

        # Extracting images
        All_mem_size = self.interval[0] + 1
        for i in reversed(range(-self.interval[1], All_mem_size * self.sample_rate, self.sample_rate)):# i = 3, 2, 1, 0
            # Labels
            if not self.baseline_mode:
                abs_file_path_gt, new_file_path_gt = self.filename_from_index(file_path_gt, i)
            else:
                abs_file_path_gt, new_file_path_gt = self.filename_from_base(file_path_gt, i)
            label = None
            orig_label = None
            new_file_path_gt = ""

            if not os.path.exists(abs_file_path_gt):
                if i == self.interval[1]:
                    print(abs_file_path_gt, "does not exist !")
                    exit(1)
            else:
                with open(abs_file_path_gt, 'rb') as f:
                    label = Image.open(f).convert('P')
                    label = Resize(self.work_res, InterpolationMode.NEAREST)(label)
                    if self.use_orig_res:
                        orig_label = Image.open(f).convert('P')

            labels.append(label) # [None, None, None, GT]
            labels_filenames.append(new_file_path_gt)
            if self.use_orig_res:
                orig_labels.append(orig_label)

            # Images
            if not self.baseline_mode:
                abs_file_path, new_file_path = self.filename_from_index(file_path, i)
            else:
                abs_file_path, new_file_path = self.filename_from_base(file_path, i)
            if not os.path.exists(abs_file_path):
                print(abs_file_path, "does not exist !")
                exit(1)
            with open(abs_file_path, 'rb') as f:
                image = Image.open(f).convert('RGB')
                image = Resize(self.work_res, InterpolationMode.BILINEAR)(image)
                if self.use_orig_res:
                    orig_img = Image.open(f).convert('RGB')

            images.append(image)    # [img3, img2, img1, ori_img]
            images_filenames.append(new_file_path)
            if self.use_orig_res:
                orig_images.append(ToTensor()(orig_img))

            # Thermal Images
            if not self.baseline_mode:
                abs_file_path_ir, new_file_path_ir = self.filename_from_index(file_path_ir, i)
            else:
                abs_file_path_ir, new_file_path_ir = self.filename_from_base(file_path_ir, i)
            if not os.path.exists(abs_file_path_ir):
                print(abs_file_path_ir, "does not exist !")
                exit(1)
            with open(abs_file_path_ir, 'rb') as f:
                thermal = Image.open(f).convert('RGB')     # load 3-channel format
                thermal = Resize(self.work_res, InterpolationMode.BILINEAR)(thermal)
                if self.use_orig_res:
                    orig_thermal = Image.open(f).convert('RGB')

            thermals.append(thermal)  # [img3, img2, img1, ori_img]
            thermals_filenames.append(new_file_path_ir)
            if self.use_orig_res:
                orig_thermals.append(ToTensor()(orig_thermal))


        # Transforming images and labels
        if self.co_transform is not None:
            images,thermals,labels = self.co_transform(images,thermals, labels)

        for i, image in enumerate(images):
            # Converting to labels
            if labels[i] is not None:
                labels[i] = self.target_transform(labels[i])
            if self.use_orig_res and orig_labels[i] is not None:
                orig_labels[i] = self.target_transform(orig_labels[i])

            if self.co_transform is None:
                images[i] = ToTensor()(image)
                thermals[i] = ToTensor()(thermals[i])


        # [img3, img2, img1, ori_img]
        # [None, None, None, ori_GT]
        images = torch.stack(images)                    # images = [seq_len, channels, h, w]
        thermals=torch.stack(thermals)
        labels = labels[self.interval[0]].unsqueeze(0)  # labels = [1, channels, h, w], the last-frame as GT
        if self.use_orig_res:
            orig_labels = orig_labels[self.interval[0]].unsqueeze(0)
            orig_images = torch.stack(orig_images)
            orig_thermals=torch.stack(orig_thermals)

        return images, thermals, labels, orig_images, orig_thermals, orig_labels, \
               file_path, file_path_gt, images_filenames, labels_filenames

    def __len__(self):
        return len(self.filenames)


    def filename_from_index(self, base_file_path, index):
        # e.g., ../MVSeg_Dataset/data/INO_ClosePerson1/label/00237l.png'
        file_name = os.path.basename(base_file_path)    # '00237l.png'
        old_num = int(file_name[-9:-5])                 # '00237' -> int = 237
        new_num = old_num - index                       # 237 - index, e.g., index=2, obtaining 235
        name_elts = "{:04d}".format(new_num)            # '00235'
        new_file_name = file_name[:-9]+name_elts+ file_name[-5:]  # new file_name

        new_path = os.path.join(base_file_path[:-len(new_file_name)], new_file_name)

        if os.path.exists(new_path):
            return new_path, new_file_name
        else:
            return base_file_path, file_name

    def filename_from_base(self, base_file_path, index):
        file_name = os.path.basename(base_file_path)
        return base_file_path, file_name

