import os
from PIL import Image

# from TC_score.RAFT_core.raft import RAFT
# from TC_score.RAFT_core.utils.utils import InputPadder
# from TC_score.metrics import Evaluator
from RAFT_core.raft import RAFT
from RAFT_core.utils.utils import InputPadder
from metrics import Evaluator

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import sys
import torch.nn.functional as F
import tqdm


# This eval code is mainly based on 'https://github.com/sssdddwww2/CVPR2021_VSPW_Implement/blob/master/TC_cal.py'
# We recommend it is only for reference,
# because in darkness or nighttime scenarios, the generation of optical flow will be inaccurate.



num_class=26


DIR_= "/set_your_path/MVSeg_Dataset/"
data_dir=DIR_+'/data'
result_dir='../save/eval_msa_deeplab/predicts'


split='test.txt'
with open(os.path.join(DIR_,split),'r') as f:
    list_ = f.readlines()
    list_ = [v.strip() for v in list_]





def flowwarp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()

    if x.is_cuda:
        grid = grid.to(x.device)
    vgrid = grid + flo

    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0

    vgrid = vgrid.permute(0,2,3,1)
    output = nn.functional.grid_sample(x, vgrid,mode='nearest',align_corners=False)

    return output





###
model_raft = RAFT()
if torch.cuda.is_available():
    to_load = torch.load('./RAFT_core/raft-things.pth-no-zip')
else:
    to_load = torch.load('./RAFT_core/raft-things.pth-no-zip',map_location='cpu')
new_state_dict = OrderedDict()
for k, v in to_load.items():
    name = k[7:]
    new_state_dict[name] = v
model_raft.load_state_dict(new_state_dict)
if torch.cuda.is_available():
    model_raft = model_raft.cuda()
else:
    model_raft = model_raft
###
total_TC=0.
evaluator = Evaluator(num_class)
for video in tqdm.tqdm(list_):
    if video[0]=='.':
        continue


    imglist_ = sorted([f for f in os.listdir(os.path.join(data_dir,video,'label'))
                if any(f.endswith(ext) for ext in ['.jpg', '.png'])])

    img_postfix =  sorted(os.listdir(os.path.join(data_dir,video,'visible')))[0][-4:]


    for i,img in enumerate(imglist_[:-1]):
        if img[0]=='.' or len(imglist_) <=1:
            continue
        #print('processing video : {} image: {}'.format(video,img))
        next_img = imglist_[i+1]
        imgname = img
        next_imgname = next_img
        img = Image.open(os.path.join(data_dir,video,'visible',img.replace('l','v')[:-4]+img_postfix))
        next_img =Image.open(os.path.join(data_dir,video,'visible',next_img.replace('l','v')[:-4]+img_postfix))
        image1 = torch.from_numpy(np.array(img))
        image2 = torch.from_numpy(np.array(next_img))
        padder = InputPadder(image1.size()[:2])
        image1 = image1.unsqueeze(0).permute(0,3,1,2)
        image2 = image2.unsqueeze(0).permute(0,3,1,2)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        if torch.cuda.is_available():
            image1 = image1.cuda()
            image2 = image2.cuda()
        else:
            image1 = image1
            image2 = image2
        with torch.no_grad():
            model_raft.eval()
            _,flow = model_raft(image1,image2,iters=20, test_mode=True)
            flow = padder.unpad(flow)

        flow = flow.data.cpu()
        pred = Image.open(os.path.join(result_dir,video,imgname.split('.')[0]+'.png'))
        next_pred = Image.open(os.path.join(result_dir,video,next_imgname.split('.')[0]+'.png'))
        pred =torch.from_numpy(np.array(pred))
        next_pred = torch.from_numpy(np.array(next_pred))
        next_pred = next_pred.unsqueeze(0).unsqueeze(0).float()


        flow = F.interpolate(flow, next_pred.size()[2:], mode='bilinear', align_corners=True)
        warp_pred = flowwarp(next_pred,flow)
    #    print(warp_pred)
        warp_pred = warp_pred.int().squeeze(1).numpy()
        pred = pred.unsqueeze(0).numpy()
        evaluator.add_batch(pred, warp_pred)
#    v_mIoU = evaluator.Mean_Intersection_over_Union()
#    total_TC+=v_mIoU
#    print('processed video : {} score:{}'.format(video,v_mIoU))

#TC = total_TC/len(list_)
TC = evaluator.Mean_Intersection_over_Union()

print("TC score is {}".format(TC))
