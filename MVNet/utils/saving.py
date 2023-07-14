import os
import torch
import numpy as np

from shutil import copyfile
from PIL import Image
from torchvision.transforms import ToPILImage
import matplotlib.pyplot as plt


class Saver:
    def __init__(self, args):

        self.baseline_path = args.baseline_path
        self.save_all_vals = args.save_all_vals
        self.save_dir = args.savedir
        self.epochs_save = args.epochs_save
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # Saving arguments
        args_save_path = os.path.join(self.save_dir, "args.txt")
        with open(args_save_path, "w") as myfile:
            myfile.write(str(args))

    def save_model_copy(self, model_path):
        copyfile(model_path, os.path.join(self.save_dir, os.path.basename(model_path)))

    def save_checkpoint(self, state_dict, is_best, enc):

        tag = "_enc" if enc else ""
        filename_ckpt = os.path.join(self.save_dir, "checkpoint{}.pth.tar".format(tag))
        filename_best = os.path.join(self.save_dir, "model_best{}.pth.tar".format(tag))

        torch.save(state_dict, filename_ckpt)
        print("Saving model checkpoint:", filename_ckpt)

        if is_best:
            print("Saving model as best:", filename_best)
            torch.save(state_dict, filename_best)

        return filename_ckpt, filename_best

    def save_model(self, model, enc, is_best, epoch, step, mean_iou):

        tag = "_encoder" if enc else ""
        filename_model = os.path.join(self.save_dir, "model{}-{:03}.pth".format(tag, epoch))
        filename_best = os.path.join(self.save_dir, "model{}_best.pth".format(tag))

        if self.epochs_save > 0 and step > 0 and step % self.epochs_save == 0:
            torch.save(model.state_dict(), filename_model)
            print("Saving for epoch:", epoch, "model:", filename_model)
        if is_best:
            torch.save(model.state_dict(), filename_best)
            print("Saving for epoch:", epoch, "model as best:", filename_best)

            text_file = os.path.join(self.save_dir, "best{}.txt".format(tag))
            with open(text_file, "w") as myfile:
                myfile.write("Best epoch is %d, with mean IoU = %.4f" % (epoch, mean_iou))

    def save_epoch_report(self, enc, epoch, mean_loss_train, mean_loss_val,
                          mean_iou_train, mean_iou_val, lr):
        tag = "_encoder" if enc else ""
        filename = os.path.join(self.save_dir, "automated_log{}.txt".format(tag))

        epoch_str = "\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % \
                    (epoch, mean_loss_train, mean_loss_val, mean_iou_train*100, mean_iou_val*100, lr)

        if not os.path.isfile(filename):
            header = "Epoch\tTrain-loss\tVal-loss\tTrain-mIoU\tVal-mIoU\tlearningRate"
        else:
            header = None

        with open(filename, "a") as f:
            # Write header only once
            if header is not None:
                f.write(header)
            f.write(epoch_str)

    def save_txtmodel(self, model, enc):
        tag = "_encoder" if enc else ""
        filename = os.path.join(self.save_dir, "model{}.txt".format(tag))
        with open(filename, "w") as f:
            f.write(str(model))


    def save_plot_group(self, dataset, seq_idx, group, iou_classes_list_list, tag):

        for i, class_info in enumerate(dataset.classes_dict):
            if class_info not in group[1]:
                continue
            plt.plot(iou_classes_list_list[i], label=class_info)
            mean_val = np.nanmean(iou_classes_list_list[i])
            median_val = np.nanmedian(iou_classes_list_list[i])
            plt.plot([], [], ' ', label="Mean diff: {:.6f}".format(mean_val))
            plt.plot([], [], ' ', label="Medn diff: {:.6f}".format(median_val))


        plt.title("Sequence {:2d} - {} - {}".format(seq_idx, os.path.basename(self.save_dir), group[0]))
        plt.grid()

        lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

        fname = os.path.join(self.save_dir, "seq_{}_{}_{}.png".format(seq_idx, group[0], tag))
        plt.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)
        plt.clf()

    def save_plot_seq_miou(self, dataset, seq_idx, iou_classes_list_list, tag):

        plt.plot(iou_classes_list_list[-1], label="miou ({} classes)".format(len(dataset.classes_dict)))
        mean_val = np.nanmean(iou_classes_list_list[-1])
        median_val = np.nanmedian(iou_classes_list_list[-1])
        min_val = np.nanmin(iou_classes_list_list[-1])
        max_val = np.nanmax(iou_classes_list_list[-1])

        plt.axhline(mean_val, color='r')
        plt.plot([], [], ' ', label="Mean diff: {:.6f}".format(mean_val))
        plt.plot([], [], ' ', label="Medn diff: {:.6f}".format(median_val))
        plt.plot([], [], ' ', label="Min  diff: {:.6f}".format(min_val))
        plt.plot([], [], ' ', label="Max  diff: {:.6f}".format(max_val))


        plt.title("Sequence {:2d} - {} - mean IoU".format(seq_idx, os.path.basename(self.save_dir)))
        plt.grid()
        if tag == "rel":
            plt.ylim(-20, 20)

        lgd = plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left")

        # plt.show()
        fname = os.path.join(self.save_dir, "seq_{}_miou_{}.png".format(seq_idx, tag))
        plt.savefig(fname, bbox_extra_artists=(lgd,), bbox_inches='tight', dpi=400)
        plt.clf()



    def save_seq_report(self, dataset, seq_idx, iou_eval_objs):

        if len(iou_eval_objs) == 0:
            return

        mean_iou_list = []
        # Len: #classes. iou_classes_list_list[k]: contains all temporal iou for that class
        iou_classes_list_list = []

        n_classes = 0

        timeline = len(iou_eval_objs)
        for t in range(timeline):
            mean_iou, iou_classes_list = iou_eval_objs[t].get_iou()
            mean_iou_list.append(mean_iou.item() * 100)
            if len(iou_classes_list_list) == 0:
                n_classes = len(iou_classes_list)
                for k in range(n_classes):
                    iou_classes_list_list.append([])
            for k in range(n_classes):
                iou_classes_list_list[k].append(iou_classes_list[k].item() * 100)

        iou_classes_list_list.append(mean_iou_list)
        summary_array = np.array(iou_classes_list_list)

        groups = {
            "landscape": ["sky", "road", "sidewalk", "building"],
            "nature": ["terrain", "tree", "vegetation"],
            "infra_big": ["infrastructure", "fence", "billboard"],
            "infra_small": ["traffic light", "traffic sign", "mobile barrier"],
            "misc": ["fire hydrant", "chair", "trash", "trashcan"],
            "dynamic": ["person", "motorcycle", "car", "van", "bus", "truck"],
        }

        for group in groups.items():
            self.save_plot_group(dataset, seq_idx, group, summary_array, "abs")
        self.save_plot_seq_miou(dataset, seq_idx, summary_array, "abs")

        if self.save_all_vals:
            fname = os.path.join(self.save_dir, "seq_{}_vals.txt".format(seq_idx))
            np.savetxt(fname, summary_array.transpose(), fmt="%6.2f")
        else:
            baseline_fname = os.path.join(self.baseline_path, "seq_{}_vals.txt".format(seq_idx))
            baseline_vals = np.loadtxt(baseline_fname)
            # print("loaded baseline values from:", baseline_fname)

            for k in range(n_classes+1): # now the miou is the last row ...
                seq_len = len(summary_array[k, :])
                summary_array[k][:] = np.subtract(summary_array[k, :], baseline_vals[:seq_len, k])
            fname = os.path.join(self.save_dir, "seq_{}_cmp.txt".format(seq_idx))
            np.savetxt(fname, summary_array.transpose(), fmt="%6.2f")

            for group in groups.items():
                self.save_plot_group(dataset, seq_idx, group, summary_array, "rel")
            self.save_plot_seq_miou(dataset, seq_idx, summary_array, "rel")



def save_image_blend(image_path, image_1, image_2, alpha):
    Image.blend(image_1, image_2, alpha).save(image_path)


def save_labels_images(dataset, savedir, idx, t, images, labels, pred_labels):

    batch_id = 1
    filename = "debug_t_{}_{}.png".format(t, idx)
    image_path = os.path.join(savedir, filename)
    img_color = ToPILImage()(images[batch_id, t, :, :, :].cpu())
    img_color_labels_gt = ToPILImage()(dataset.colorize(labels[batch_id, t, :, :, :].cpu()))
    img_color_labels_pred = ToPILImage()(dataset.colorize(pred_labels[batch_id, t, :, :, :].cpu()))

    # Making sure it exists
    os.makedirs(os.path.dirname(image_path), exist_ok=True)

    # Saving blends
    save_image_blend(image_path.replace("debug", "debug_gt".format(idx)),
                     img_color, img_color_labels_gt, 0.5)
    save_image_blend(image_path.replace("debug", "debug_pred".format(idx)),
                     img_color, img_color_labels_pred, 0.5)
