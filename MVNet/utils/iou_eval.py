import torch
import numpy as np

class IouEval:

    def __init__(self, n_classes, ignore_idx=-1):
        self.n_classes = n_classes
        # The ignore index is the last one
        self.ignoreIndex = ignore_idx #n_classes - 1

        classes = self.n_classes if self.ignoreIndex == -1 else self.n_classes - 1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()

    def add_batch(self, preds, labels):
        # sizes should be "batch_size x nClasses x H x W"

        if preds.is_cuda or labels.is_cuda:
            preds = preds.cuda()
            labels = labels.cuda()

        # if size is "batch_size x 1 x H x W" scatter to onehot
        if preds.size(1) == 1:
            x_onehot = torch.zeros(preds.size(0), self.n_classes, preds.size(2), preds.size(3))
            if preds.is_cuda:
                x_onehot = x_onehot.cuda()
            x_onehot.scatter_(1, preds, 1).float()
        else:
            x_onehot = preds.float()

        if labels.size(1) == 1:
            y_onehot = torch.zeros(labels.size(0), self.n_classes, labels.size(2), labels.size(3))
            if labels.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, labels, 1).float()
        else:
            y_onehot = labels.float()

        if self.ignoreIndex != -1:
            ignores = y_onehot[:, self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores = 0


        tpmult = x_onehot * y_onehot
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True),
                       dim=3, keepdim=True).squeeze()
        # times pred says its that and gt says its not (subtract cases when its ignore label!)
        fpmult = x_onehot * (1 - y_onehot - ignores)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True),
                       dim=3, keepdim=True).squeeze()
        # times pred says its not that and gt says it is
        fnmult = (1 - x_onehot) * y_onehot
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True),
                       dim=3, keepdim=True).squeeze()

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def get_iou(self):
        smooth = 1e-15
        num = self.tp
        den = self.tp + self.fp + self.fn
        iou = (num + smooth) / (den + smooth)

        # return torch.mean(iou), iou
        return np.nanmean(iou.numpy()), iou




