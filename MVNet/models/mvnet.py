import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import utils.visualize

from models.deeplabv3_plus import DeepLabPredict
from models.deeplabv3_plus import DeepLabv3plus_backbone
from datasets.helpers import DATASETS_NUM_CLASSES


class MemoryCrossmodality(nn.Module):
    def __init__(self, args, in_dim=256):
        super(MemoryCrossmodality, self).__init__()
        self.TT = args.stm_queue_size
        self.conv_layer = nn.Conv2d(in_dim * self.TT, in_dim, kernel_size=1)
        self.conv_fusion = nn.Conv2d(in_dim * 2, in_dim, kernel_size=1)
        self.temperature = 1.0

    def Crossmodal_fusion(self, model_proto, feat):    # model_proto: (N, T, C) Feat: (N, H, W)
        n, h, w = feat.shape
        feat = feat.view(n, h * w).permute(1, 0).contiguous()                           # Feat: HW * N
        feat_norm = F.normalize(feat, dim=-1)
        n_p, n_T, n_C = model_proto.shape
        model_proto = model_proto.view(n_p, n_T * n_C).contiguous()
        prototypes_norm = F.normalize(model_proto, dim=0)                               # Proto: N * TC
        attn = torch.mm(feat_norm, prototypes_norm) / self.temperature                  # Attn: HW * TC
        attn = torch.softmax(attn, dim=-1)
        model_proto = model_proto.permute(1, 0).contiguous()                            # Transpose_Proto: TC * N
        feat_re = torch.mm(attn, model_proto)                                           # feat_re: HW * N
        feat_re = feat_re.view(h, w, n).permute(2, 0, 1).contiguous().unsqueeze(0)      # feat_re: B, N, H ,W
        return feat_re

    def forward(self, feat_seq, feat):
        B, Dim, T, C = feat_seq.size()
        B, Dim, H, W = feat.size()
        ori_feat = feat

        feat_batch = []
        for j in range(B):
            feat_seq_j = feat_seq[j, ...]   # N * T * C
            feat_j = feat[j, ...]           # N * H * W

            fused_feat_j = self.Crossmodal_fusion(feat_seq_j, feat_j)
            feat_batch.append(fused_feat_j)

        mem_feats = torch.cat(feat_batch, dim=0)

        # Fusion
        fused_fea = self.conv_fusion(torch.cat((mem_feats, ori_feat), dim=1))

        return fused_fea, mem_feats


class MemoryQueue_Prototype():

    def __init__(self, args):
        self.queue_size = args.stm_queue_size
        self.queue_vals = []
        self.queue_idxs = []

    def reset(self):
        self.queue_vals = []
        self.queue_idxs = []

    def current_size(self):
        return len(self.queue_vals)

    def update(self, val, idx):
        self.queue_vals.append(val)
        self.queue_idxs.append(idx)

        if len(self.queue_vals) > self.queue_size:
            self.queue_vals.pop(0)
            self.queue_idxs.pop(0)

    def get_indices(self):
        return self.queue_idxs

    def get_vals(self):
        return torch.stack(self.queue_vals, dim=2)


class Aggregate_layer(nn.Module):
    # Not using location
    def __init__(self, in_dim):
        super(Aggregate_layer, self).__init__()
        self.conv_layer = nn.Conv2d(in_dim * 3, in_dim, kernel_size=1)

    def forward(self, f_rgb, f_thermal, f_fusion):
        feature = torch.cat((f_rgb, f_thermal, f_fusion), dim=1)
        return self.conv_layer(feature)


class MVNet(nn.Module):
    def __init__(self, args, encoder=None, nobn=False, print_all_logs=True, board=None):
        super(MVNet, self).__init__()
        self.num_classes = DATASETS_NUM_CLASSES[args.dataset]       # Semantic classes
        self.is_training = args.training                            # Training Mode
        self.model_struct = args.model_struct                       # 'original' by default
        self.baseline_mode = args.baseline_mode                     # store_true
        self.always_decode = args.always_decode
        self.memory_strategy = args.memory_strategy                 # sigmoid-do1
        self.stm_queue_size = args.stm_queue_size                   # 3
        self.sample_rate = args.sample_rate
        self.backbone = args.backbone                               # deeplab

        if args.backbone == "deeplab50":
            use_layer = 50
            self.encoder = DeepLabv3plus_backbone(pretrained=False, layers=use_layer)
            self.decoder = DeepLabPredict(self.num_classes)
            fea_dim = 256
            if self.model_struct == "original":
                # Nothing else to do
                pass
            else:
                print("model_struct Not implemented")
                exit(1)
        else:
            print("Unknown backbone")
            exit(1)

        self.memory_module_crossmodality_R = MemoryCrossmodality(args, in_dim=fea_dim)
        self.memory_module_crossmodality_T = MemoryCrossmodality(args, in_dim=fea_dim)
        self.memory_module_crossmodality_F = MemoryCrossmodality(args, in_dim=fea_dim)
        self.memory_queue = MemoryQueue_Prototype(args)
        self.memory_queue_thermal = MemoryQueue_Prototype(args)
        self.memory_queue_fusion = MemoryQueue_Prototype(args)

        self.aggregate_R = Aggregate_layer(fea_dim)
        self.aggregate_T = Aggregate_layer(fea_dim)
        self.aggregate_F = Aggregate_layer(fea_dim)


        if print_all_logs:
            utils.visualize.print_summary(self.__dict__, self.__class__.__name__)

    def gen_prototypes(self, feat, logit, cls_num):
        n, c, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        logit = logit.permute(0, 2, 3, 1).contiguous().view(n, h * w, -1)
        label = torch.argmax(logit, dim=-1)
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

    def reset_hidden_state(self):
        self.memory_queue.reset()
        self.memory_queue_thermal.reset()
        self.memory_queue_fusion.reset()

    def memory_range(self, seq_len):
        if self.memory_strategy == "all":
            ret_range = range(seq_len)
        if self.memory_strategy == "random":
            ret_range = random.sample(range(seq_len-1), self.stm_queue_size-1)
            ret_range.append(seq_len - 1)   # ensure the last frame is label
            ret_range.sort()

        # assert(len(ret_range) == self.stm_queue_size)

        assert(seq_len - 1 in ret_range)    # image_list = [img3, img2, img1, img_with_GT]
        return ret_range



    def forward(self, input, thermal, step=0, epoch=0):
        # Input has to be of size (batch_size, seq_len, channels, h, w)
        seq_len = input.size(1)
        # If seq_len == 1, are evaluating, otherwise we are probably in training mode
        decoder_outputs = []
        decoder_outputs_aux_rgb = []
        decoder_outputs_aux_thermal = []
        decoder_outputs_aux_fusion = []
        memory_range = self.memory_range(seq_len)



        for t in memory_range:  # [0, 1, 2, 3]
            input_size = input[:, t, :, :, :].size()
            h = int(input_size[2])
            w = int(input_size[3])

            # Backbone Feature Extraction
            encoder_output, encoder_output_thermal, encoder_output_fusion = \
                    self.encoder(input[:, t, :, :, :], thermal[:, t, :, :, :])


            if self.baseline_mode:  # w/o memory information
                if not (self.always_decode or (t == seq_len - 1)):
                    continue
                output_decoded, aux_RGB, aux_thermal, aux_fusion = \
                    self.decoder(encoder_output,encoder_output_thermal,encoder_output_fusion, h, w)
                decoder_outputs.append(output_decoded)
                decoder_outputs_aux_rgb.append(aux_RGB)
                decoder_outputs_aux_thermal.append(aux_thermal)
                decoder_outputs_aux_fusion.append(aux_fusion)
                continue




            '''When frame is the last frame，i.e. t == seq_len - 1，performing proto-STM with memory'''
            if self.always_decode or (t == seq_len - 1):
                if self.memory_queue.current_size() != 0:

                    # RGB to R\T\F
                    fused_mem, _ = self.memory_module_crossmodality_R(self.memory_queue.get_vals(), encoder_output)
                    fused_mem_r_t, _ = self.memory_module_crossmodality_R(self.memory_queue_thermal.get_vals(), encoder_output)
                    fused_mem_r_f, _= self.memory_module_crossmodality_R(self.memory_queue_fusion.get_vals(), encoder_output)
                    fused_mem_R = self.aggregate_R(fused_mem, fused_mem_r_t, fused_mem_r_f)

                    # Thermal to R\T\F
                    fused_mem_t_r, _ = self.memory_module_crossmodality_T(self.memory_queue.get_vals(), encoder_output_thermal)
                    fused_mem_t_t, _ = self.memory_module_crossmodality_T(self.memory_queue_thermal.get_vals(), encoder_output_thermal)
                    fused_mem_t_f, _ = self.memory_module_crossmodality_T(self.memory_queue_fusion.get_vals(), encoder_output_thermal)
                    fused_mem_T = self.aggregate_T(fused_mem_t_r, fused_mem_t_t, fused_mem_t_f)

                    # Fusion to R\T\F
                    fused_mem_f_r, _ = self.memory_module_crossmodality_F(self.memory_queue.get_vals(), encoder_output_fusion)
                    fused_mem_f_t, _ = self.memory_module_crossmodality_F(self.memory_queue_thermal.get_vals(), encoder_output_fusion)
                    fused_mem_f_f, _ = self.memory_module_crossmodality_F(self.memory_queue_fusion.get_vals(), encoder_output_fusion)
                    fused_mem_F = self.aggregate_F(fused_mem_f_r, fused_mem_f_t, fused_mem_f_f)


                    # Construct Set for metric learning
                    Current_fea = [encoder_output, encoder_output_thermal, encoder_output_fusion]
                    Previous_mem= [self.memory_queue.get_vals(), self.memory_queue_thermal.get_vals(),
                                   self.memory_queue_fusion.get_vals()]


                    if not (self.always_decode or (t == seq_len - 1)):
                        continue


                    if self.model_struct == "original":
                        output_decoded, aux_RGB, aux_thermal, aux_fusion = self.decoder(fused_mem_R, fused_mem_T, fused_mem_F, h, w)
                        decoder_outputs.append(output_decoded)
                        decoder_outputs_aux_rgb.append(aux_RGB)
                        decoder_outputs_aux_thermal.append(aux_thermal)
                        decoder_outputs_aux_fusion.append(aux_fusion)


                else:
                    # First case without memory, simply predict
                    # torch.Size([4, 20, 64, 128])
                    # torch.Size([4, 20, 512, 1024])

                    if not (self.always_decode or (t == seq_len - 1)):
                        continue

                    output_decoded, aux_RGB, aux_thermal, aux_fusion=self.decoder(encoder_output,encoder_output_thermal,encoder_output_fusion,h,w)
                    decoder_outputs.append(output_decoded)
                    decoder_outputs_aux_rgb.append(aux_RGB)
                    decoder_outputs_aux_thermal.append(aux_thermal)
                    decoder_outputs_aux_fusion.append(aux_fusion)


            # Memorize
            if self.model_struct == "original":
                _, aux_RGB_o, aux_thermal_o, aux_fusion_o = self.decoder(encoder_output,
                                                                           encoder_output_thermal,
                                                                           encoder_output_fusion, int(h/4), int(w/4))
                vM = self.gen_prototypes(encoder_output, aux_RGB_o, self.num_classes)
                vM_thermal = self.gen_prototypes(encoder_output_thermal, aux_thermal_o, self.num_classes)
                vM_fusion = self.gen_prototypes(encoder_output_fusion, aux_fusion_o, self.num_classes)

                idx = step if seq_len == 1 else t
                self.memory_queue.update(vM, idx)
                self.memory_queue_thermal.update(vM_thermal, idx)
                self.memory_queue_fusion.update(vM_fusion, idx)



        output_decoder = torch.stack(decoder_outputs, dim=1)
        outputs_aux_rgb, outputs_aux_thermal, outputs_aux_fusion = None, None, None
        if decoder_outputs_aux_rgb:
            outputs_aux_rgb = torch.stack(decoder_outputs_aux_rgb, dim=1)
        if decoder_outputs_aux_thermal:
            outputs_aux_thermal = torch.stack(decoder_outputs_aux_thermal, dim=1)
        if decoder_outputs_aux_fusion:
            outputs_aux_fusion = torch.stack(decoder_outputs_aux_fusion, dim=1)

        if self.baseline_mode:
            Total_feas = None
        else:
            Total_feas = [Current_fea, Previous_mem]

        return output_decoder, outputs_aux_rgb, outputs_aux_thermal, outputs_aux_fusion, Total_feas




if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--dataset', default="MVSeg")
    parser.add_argument('--training', action='store_true')
    parser.add_argument('--model', default="mvnet.py")
    parser.add_argument('--always-decode', action='store_true')
    parser.add_argument('--backbone', type=str, default="deeplab50")
    parser.add_argument('--baseline-mode', action='store_true', default=False)
    parser.add_argument('--win-size', type=int, default=-1, help='if not given, mem size + 1')
    parser.add_argument('--stm-queue-size', type=int, default=3)
    parser.add_argument('--model-struct', type=str, default="original")
    parser.add_argument('--memorize-first', action='store_true')
    parser.add_argument('--memory-strategy', type=str, default="all")
    args = parser.parse_args()


    from thop import profile
    model = MVNet(args)

    image = torch.randn(1, 4, 3, 320, 480)
    thermal = torch.randn(1, 4, 3, 320, 480)

    flops, params = profile(model, inputs=(image, thermal))


    print('Params: %.1f (M)' % (params / 1000000.0))
    print('FLOPS: %.1f (G)' % (flops / 1000000000.0))



