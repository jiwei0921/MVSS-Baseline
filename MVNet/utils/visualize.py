import numpy as np
import re
import os
import statistics

from torch.autograd import Variable
from pathlib import Path
from shutil import copyfile

from visdom import Visdom
import torch
import utils.visdom_helpers as vis_helpers

class Dashboard:

    def __init__(self, args):
        self.vis = Visdom(port=args.port)
        self.cfg_name = os.path.basename(os.path.normpath(args.savedir))
        self.envs = set()

        self.debug = 10
        self.registered_blocks = {}
        self.blocks_list = []

        self.vis.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')
        self.vis.register_event_handler(self.block_list_callback_handler, 'block_list')

    def block_list_callback_handler(self, data):
        field_name = self.blocks_list[data['propertyId']]['name']

        self.registered_blocks[field_name].toggle_display(data['value'])

        self.blocks_list[data['propertyId']]['value'] = data['value']

        self.vis.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

    def register(self, data, mode, debug_level=0, title='Data', **kwargs):
        if title not in self.registered_blocks.keys():
            show_data = self.debug >= debug_level

            if title != 'Tracking':
                self.blocks_list.append({'type': 'checkbox', 'name': title, 'value': show_data})

            self.vis.properties(self.blocks_list, opts={'title': 'Block List'}, win='block_list')

            if mode == 'image':
                self.registered_blocks[title] = vis_helpers.VisImage(self.vis, show_data, title)
            elif mode == 'heatmap':
                self.registered_blocks[title] = vis_helpers.VisHeatmap(self.vis, show_data, title)
            elif mode == 'cost_volume':
                self.registered_blocks[title] = vis_helpers.VisCostVolume(self.vis, show_data, title)
            elif mode == 'cost_volume_flip':
                self.registered_blocks[title] = vis_helpers.VisCostVolume(self.vis, show_data, title, flip=True)
            elif mode == 'cost_volume_ui':
                self.registered_blocks[title] = vis_helpers.VisCostVolumeUI(self.vis, show_data, title, data[1],
                                                                self.registered_blocks)
            elif mode == 'info_dict':
                self.registered_blocks[title] = vis_helpers.VisInfoDict(self.vis, show_data, title)
            elif mode == 'text':
                self.registered_blocks[title] = vis_helpers.VisText(self.vis, show_data, title)
            elif mode == 'lineplot':
                self.registered_blocks[title] = vis_helpers.VisLinePlot(self.vis, show_data, title)
            elif mode == 'featmap':
                self.registered_blocks[title] = vis_helpers.VisFeaturemap(self.vis, show_data, title)
            else:
                raise ValueError('Visdom Error: Unknown data mode {}'.format(mode))
        # Update
        self.registered_blocks[title].update(data, **kwargs)


    def save_dashboard(self, save_dir):
        print("Saving dashboard envs:", self.envs)
        self.vis.save(list(self.envs))

        for env in self.envs:
            env_src = os.path.join(str(Path.home()), ".visdom", env + ".json")
            env_dst = os.path.join(save_dir, env + ".json")
            copyfile(env_src, env_dst)


    def loss(self, losses, title):
        x = np.arange(1, len(losses)+1, 1)

        self.vis.line(losses, x, env='loss', opts=dict(title=title))

    def image(self, image, title):
        if image.is_cuda:
            image = image.cpu()
        if isinstance(image, Variable):
            image = image.data
        image = image.numpy()

        self.vis.image(image, env='images', win=title, opts=dict(title=title, store_history=True))

    """
    runs during training only
    """
    def update_board_images(self, images, pred_labels, labels, epoch, step, subset_tag, shallow_dec, colorize):

        if step != 0:
            return

        if subset_tag == "TRAIN": # Saving complexity for the browser
            return

        win_size = pred_labels.size(1)
        n_imgs_row = 4 # number of images per row

        # To do during first epoch only for VALIDATION
        store_hist = subset_tag == "TRAIN"
        print_target_once = subset_tag == "TRAIN" or (subset_tag == "VAL" and epoch == 1)

        colored_labels = []
        colored_target = []
        for t in range(win_size):
            colored_labels.append(colorize(pred_labels[0, t, :, :, :]))
            if print_target_once and t < labels.size(1):
                colored_target.append(colorize(labels[0, t, :, :, :]))

        colored_labels = torch.stack(colored_labels)
        if colored_target:
            colored_target = torch.stack(colored_target)

        env_name = "{} {}".format("shallow_decoder" if shallow_dec else "full_decoder", self.cfg_name)
        self.envs.add(env_name)

        inputs_str = "Sequence {} INPUTS".format(subset_tag, step)
        labels_str = "Sequence {} LABELS".format(subset_tag, step)
        target_str = "Sequence {} TARGET".format(subset_tag, step)

        if print_target_once:
            self.vis.images(images[0, :, :, :, :], env=env_name, win=inputs_str, nrow=n_imgs_row,
                            opts=dict(title=inputs_str, store_history=store_hist))
            self.vis.images(colored_target, env=env_name, win=target_str, nrow=n_imgs_row,
                            opts=dict(title=target_str, store_history=store_hist))
        self.vis.images(colored_labels, env=env_name, win=labels_str, nrow=n_imgs_row,
                        opts=dict(title=labels_str, store_history=True))

    def update_board_images_grid(self, input_images, probs, labels, epoch, step, subset_tag):
        self.vis.images(input_images, env='images', win="inputs", opts=dict(title="inputs", nrow=1))

    def update_graphs(self, shallow_dec, epoch, mean_loss_train, mean_loss_val, mean_iou_val, usedLr):

        env_name = "graphs {}".format(self.cfg_name)
        self.envs.add(env_name)

        tag = "shallow decoder" if shallow_dec else "full decoder"
        str_1 = "mLoss TRAIN ({})".format(tag)
        str_2 = "mLoss VAL ({})".format(tag)
        str_4 = "mIoU VAL ({})".format(tag)
        str_5 = "L.R. ({})".format(tag)

        # env_exists = os.path.exists(os.path.join(str(Path.home()), ".visdom", env_name + ".json"))
        # mode = 'append' if env_exists else ''
        # TODO for some reason upon resume, this does not work
        mode = '' if epoch == 1 else 'append'

        self.vis.line([usedLr], [epoch], env=env_name, win=str_5,
                      opts=dict(title=str_5), update=mode, name=self.cfg_name)
        self.vis.line([mean_loss_train], [epoch], env=env_name, win=str_1,
                      opts=dict(title=str_1), update=mode, name=self.cfg_name)
        self.vis.line([mean_loss_val], [epoch], env=env_name, win=str_2,
                      opts=dict(title=str_2), update=mode, name=self.cfg_name)
        self.vis.line([mean_iou_val], [epoch], env=env_name, win=str_4,
                      opts=dict(title=str_4), update=mode, name=self.cfg_name)

    def update_correlation(self, correlation_tensor, image, correlation_local=None):

        correlation_tensor_cpu = correlation_tensor.cpu()
        correlation_tensor_cpu_l = correlation_local.cpu()

        print("Updating correlation tensors")
        for i in range(0, correlation_tensor.shape[0]):

            print("Updating correlation tensors t = {}".format(i))
            self.register(correlation_tensor_cpu[i, :, :, :, :], 'cost_volume_flip', 100,   "Memory P[t={}]".format(i))
            self.register(correlation_tensor_cpu_l[i, :, :, :, :], 'cost_volume', 100, "MemLoc P[t={}]".format(i))

        self.register((image.cpu(), (64, 128)), 'cost_volume_ui', 2, 'CostVolumeUI')

        print("updated correlation visualization")


# Class for colors
class Colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'


# Colored value output if colorized flag is activated.
def get_color_entry(val):
    if val < .20:
        return Colors.RED
    elif val < .40:
        return Colors.YELLOW
    elif val < .60:
        return Colors.BLUE
    elif val < .80:
        return Colors.CYAN
    else:
        return Colors.GREEN


def colored_str(score, msg):
    return get_color_entry(score) + msg + Colors.ENDC


def print_summary(data_dict, name):

    print("===============================================================")
    print(name, "SUMMARY")

    # Get the longest subject name
    length = max(len(x) for x in data_dict)

    # format and print the statement
    for key, value in data_dict.items():
        # Do not display private variables (dicts usually)
        # Do not display file names
        # Do not display nn.Module "training" field, always true and can be confusing
        if re.match(r"_.*", key) or \
                re.match(r"filenames.*", key) or \
                re.match(r".*transform", key) or \
                re.match(r"classes_dict", key) or \
                key == "training":
            continue
        if re.match(r".*root", key):
            if not os.path.exists(value):
                print("Error! Provided root dir does not exist or cannot be read:", value)

        subject = "{0:>{space}}".format(key, space=length + 2)
        result = ": {0:}".format(value)
        print(subject + result)

def print_classes_iou(classes_dict, iou_classes, iou_classes_orig=None):

    print("Per-Class IoU (WorkRes{}):".format("|OrigRes" if iou_classes_orig is not None else ""))
    for i, class_info in enumerate(classes_dict):
        seq_ious_str = ""
        for t in range(len(iou_classes)):
            iou1 = colored_str(iou_classes[t][i], '{:5.2f}'.format(iou_classes[t][i]*100))
            seq_ious_str += iou1 + " % "
            if iou_classes_orig is not None:
                iou2 = colored_str(iou_classes_orig[t][i], '{:5.2f}'.format(iou_classes_orig[t][i]*100))
                seq_ious_str += "| " + iou2 + " % "
        print(class_info.rjust(18), ":", seq_ious_str)
        # Last element is the ignore class
        if i == len(classes_dict) - 2:
            break


def print_iou_report(classes_dict, iou_val, iou_classes, iou_val_orig, iou_classes_orig, total_time):

    print("---------------------------------------")
    print("Took ", total_time, "seconds")
    print("=======================================")
    print_classes_iou(classes_dict, iou_classes, iou_classes_orig)
    print("=======================================")
    miou_str1 = colored_str(iou_val, '{:0.2f}'.format(iou_val * 100))
    miou_str2 = colored_str(iou_val_orig, '{:0.2f}'.format(iou_val_orig * 100))
    print("MEAN IoU work res: " + miou_str1 + " % | Orig res: " + miou_str2 + " %")


def print_optimized_model_params(model):
    print("Optimized parameters:")
    count = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name)
            count += 1
    print("Total optimized parameters:", count)


def print_classes_report(dataset, iou_eval_objs, tag_short, is_train):
    all_mious = ""
    mious = []
    iou_classes_list = []
    for t in range(len(iou_eval_objs)):
        mean_iou, iou_classes = iou_eval_objs[t].get_iou()
        iou_str = colored_str(mean_iou, '{:6.2f}'.format(mean_iou * 100))
        all_mious += iou_str + " %"
        mious.append(mean_iou.item())
        iou_classes_list.append(iou_classes)
    seq_miou = statistics.mean(mious)
    seq_miou_str = colored_str(seq_miou, '{:6.2f}'.format(seq_miou * 100)) + " %"
    print("EPOCH on", tag_short, "set: Seq mIoU", seq_miou_str)
    print("mIoU(t) on", tag_short, "set:", all_mious)
    if not is_train:
        print_classes_iou(dataset.classes_dict, iou_classes_list)

    return seq_miou


def print_seq_report(curr_seq_idx, curr_seq_count, tag_short, seq_losses, seq_times):
    print("SEQ {:2d} : {} loss: {:.4f} | "
          "Total images {:5d} | "
          "Median time per image {:.4f} ms"
          .format(curr_seq_idx, tag_short, sum(seq_losses) / len(seq_losses),
                  curr_seq_count, 1000 * statistics.median(seq_times)))


def print_seq_overall_report(tag_short, overall_mean_loss, total_img_count, overall_times):
    report_str = "{} Overall loss: {:.4f} | Total images {:5d} | Median time per image {:.4f} ms" \
        .format(tag_short, overall_mean_loss, total_img_count, 1000 * statistics.median(overall_times))
    print(report_str)
