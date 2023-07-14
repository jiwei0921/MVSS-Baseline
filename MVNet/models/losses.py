import torch
from datasets.helpers import classes_weights, DATASETS_CLASSES_DICT, DATASETS_NUM_CLASSES

import torch.nn.functional as F


class TrainingLoss(torch.nn.Module):

    def __init__(self, args, gpu):
        super().__init__()

        weights = classes_weights(DATASETS_CLASSES_DICT[args.dataset], args.shallow_dec, gpu)

        self.win_size = args.win_size
        self.always_decode = args.always_decode

        if args.dataset == "MVSeg":
            #n_losses = 1
            n_losses = 4
            self.last_frame_loss = True
            self.idx_start = self.win_size - 1
        else:
            n_losses = -1
            print("unsupported dataset")
            exit(1)

        self.loss_idx_step = 1
        if args.model_struct == "aux_mem_loss":
            n_losses *= 2
            self.loss_idx_step = 2

        self.losses = [torch.nn.NLLLoss(weights) for i in range(n_losses)]

        assert(len(self.losses) == n_losses)
        assert(self.idx_start < self.win_size)


    def gen_prototypes(self, feat, label, cls_num):
        n, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)                 # B, H*W, N
        label = label.permute(0, 1, 2).contiguous().view(n, h * w)                      # B, H*W

        prototypes_batch = []
        for i in range(n):
            # classes = torch.unique(label[i, :].clone().detach())
            classes = list(range(cls_num))
            prototypes = []
            for c in classes:
                prototype = feat[label == c, :]
                temp = prototype.detach()
                if torch.equal(temp.cpu(), torch.zeros(prototype.shape)):
                    prototype = prototype.sum(0, keepdims=True)
                else:
                    prototype = prototype.mean(0, keepdims=True)
                prototypes.append(prototype)
            prototypes = torch.cat(prototypes, dim=0)
            prototypes = prototypes.permute(1, 0).contiguous()
            prototypes_batch.append(prototypes)
        prototypes = torch.stack(prototypes_batch)
        return prototypes  # [batch_size, N_channels, Classes]

    def metric_learning(self,curr_R_fea, mem_feats, labels):
        batch_size = curr_R_fea.shape[0]
        labels = labels.contiguous().view(batch_size, -1)  # B_s, H*W
        curr_R_fea = curr_R_fea.permute(0, 2, 3, 1)
        curr_R_fea = curr_R_fea.contiguous().view(curr_R_fea.shape[0], -1, curr_R_fea.shape[-1])  # B_s, H*W, D

        [mem_R_feas, mem_T_feas, mem_F_feas] = mem_feats

        temperature = 0.1
        loss_batch = []
        for i in range(batch_size):
            label_i = labels[i, ...]  # HW

            mem_R_feas_i = mem_R_feas[i, ...]  # D, T, C
            mem_T_feas_i = mem_T_feas[i, ...]  # D, T, C
            mem_F_feas_i = mem_F_feas[i, ...]  # D, T, C

            multimodal_mem_i = torch.cat([mem_R_feas_i, mem_T_feas_i, mem_F_feas_i], dim=1).permute(1, 2,
                                                                                                    0)  # 3T, C,  D
            T3, C, N = multimodal_mem_i.size()
            multimodal_mem_i_view = multimodal_mem_i.contiguous().view(T3 * C, N)  # 3T*C, D

            anchor_fea_R_i = curr_R_fea[i, ...]  # HW, D

            '''Takcing anchor R features as an example'''
            # anchor_fea_R_i： （HW, N）；  multimodal_mem_i：（3TC, N）
            # memory label
            y_contrast = torch.zeros((T3 * C, 1)).float().cuda()         # 3TC, 1
            sample_ptr = 0
            for ii in range(C):
                # if ii == 0: continue
                y_contrast[sample_ptr:sample_ptr + T3, ...] = ii
                sample_ptr += T3
            contrast_feature = F.normalize(multimodal_mem_i_view, p=2, dim=1)  # 3TC, D

            # valid_mask = torch.norm(contrast_feature, p=1, dim=1).cuda()   # 3TC
            # valid_mask = torch.where(valid_mask > 0, torch.tensor(1).cuda() , torch.tensor(0).cuda()).contiguous().view(-1, 1)

            y_anchor = label_i.contiguous().view(-1, 1)  # HW,  1
            anchor_feature = F.normalize(anchor_fea_R_i, p=2, dim=1)  # HW,  D

            mask = torch.eq(y_anchor, y_contrast.T).float().cuda()             # HW, 3TC
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T),
                                            temperature)  # HW, 3TC
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()  # HW, 3TC

            neg_mask = 1 - mask
            neg_mask = neg_mask #* valid_mask.T
            mask = mask #* valid_mask.T

            neg_logits = torch.exp(logits) * neg_mask
            neg_logits = neg_logits.sum(1, keepdim=True)

            exp_logits = torch.exp(logits)

            log_prob = logits - torch.log(exp_logits + neg_logits)

            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            loss = - mean_log_prob_pos
            loss = loss.mean()
            loss_batch.append(loss)


        loss_R = sum(loss_batch) / len(loss_batch)
        return loss_R


    def contrastive_loss(self, Current_fea, Previous_mem, Label):
        [curr_R_fea, curr_T_fea, curr_F_fea] = Current_fea      # Each Dim is: [B, N, H, W]
        [mem_R_feas, mem_T_feas, mem_F_feas] = Previous_mem     # Each Dim is: [B, N, T, C]

        labels = Label.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels, (curr_R_fea.shape[2], curr_R_fea.shape[3]), mode='nearest')
        labels = labels.squeeze(1).long()                       # Each Dim is: [B, H, W]

        # Update Memory
        curr_R_proto, curr_T_proto, curr_F_proto = \
            self.gen_prototypes(curr_R_fea, labels, DATASETS_NUM_CLASSES["MVSeg"]).unsqueeze(2), \
            self.gen_prototypes(curr_T_fea, labels, DATASETS_NUM_CLASSES["MVSeg"]).unsqueeze(2),\
            self.gen_prototypes(curr_F_fea, labels, DATASETS_NUM_CLASSES["MVSeg"]).unsqueeze(2)
        mem_R_feas, mem_T_feas, mem_F_feas = torch.cat([mem_R_feas, curr_R_proto], dim=2), \
                                             torch.cat([mem_T_feas, curr_T_proto], dim=2), \
                                             torch.cat([mem_F_feas, curr_F_proto], dim=2)
        mem_feats_update = [mem_R_feas, mem_T_feas, mem_F_feas]

        lossR = self.metric_learning(curr_R_fea, mem_feats_update, labels)
        lossT = self.metric_learning(curr_T_fea, mem_feats_update, labels)
        lossF = self.metric_learning(curr_F_fea, mem_feats_update, labels)

        loss_metric = (lossR + lossT + lossF)/3.0
        return loss_metric


    def forward(self, probs, labels, probs_aux_rgb, probs_thermal, probs_fusion, total_feas):
        # Probs:    torch.Size([2, 1, 26, 480, 640])
        # Labels:   torch.Size([2, 1, 1, 480, 640])

        if self.always_decode:
            assert(probs.size(1) == self.win_size)
        else:
            assert(probs.size(1) == 1)

        losses = []
        softmax_dim = 1

        # win_size = 4, idx_start = 1.

        for t in range(self.idx_start, self.win_size, self.loss_idx_step):
            label_idx = 0 if self.last_frame_loss else t
            probs_idx = 0 if not self.always_decode else t
            losses.append(self.losses[t - self.idx_start]
                          (F.log_softmax(probs[:,probs_idx,:,:,:], dim=softmax_dim), labels[:,label_idx,0,:,:]))

            if probs_aux_rgb is not None:
                losses.append(self.losses[t - self.idx_start + 1]
                           (F.log_softmax(probs_aux_rgb[:,probs_idx,:,:,:],dim=softmax_dim),labels[:,label_idx,0,:,:]))
            if probs_thermal is not None:
                losses.append(self.losses[t - self.idx_start + 2]
                           (F.log_softmax(probs_thermal[:,probs_idx,:,:,:],dim=softmax_dim),labels[:,label_idx,0,:,:]))
            if probs_fusion is not None:
                losses.append(self.losses[t - self.idx_start + 3]
                           (F.log_softmax(probs_fusion[:,probs_idx,:,:,:],dim=softmax_dim),labels[:,label_idx,0,:,:]))

        loss_ce = sum(losses) / len(losses)

        if total_feas is None:
            return loss_ce
        else:
            # Calculating the MVRegulator Loss
            [Current_fea, Previous_mem] = total_feas
            label_index = 0 if self.last_frame_loss else exit(1)
            Curr_Target = labels[:,label_index, 0,:,:]              # labels: [B, T=1, 1, H, W]

            loss_metirc = self.contrastive_loss(Current_fea, Previous_mem, Curr_Target)
            # print(loss_metirc)
            return loss_ce + 0.001 * loss_metirc
