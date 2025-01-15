import os
import json
import argparse
from pathlib import Path

import h5py
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def interp_t(tensor, T_target, mode='nearest'):
    # tensor: [T_source, D]
    D, dtype = tensor.shape[-1], tensor.dtype
    return F.interpolate(tensor[None, None].float(), size=(T_target, D), mode=mode).squeeze([0, 1]).to(dtype=dtype)

class NLQDataset(torch.utils.data.Dataset):
    def __init__(self, feature_type, caption_type='videorecap'):
        self.train_annotations = json.loads(Path(f"/data/soyeonhong/nlq/nlq_lightning/data/unified/annotations.NLQ_train.json").read_text())
        self.val_annotations = json.loads(Path(f"/data/soyeonhong/nlq/nlq_lightning/data/unified/annotations.NLQ_val.json").read_text())
        self.annotations = self.train_annotations + self.val_annotations
        self.feature_type = feature_type
        self.caption_type = caption_type
        self.video_features = h5py.File(os.path.join('/data/soyeonhong/nlq/nlq_lightning/data/unified/egovlp_internvideo.hdf5'), 'r')

        if self.feature_type == 'clip':
            # self.p_feature = Path('/data/soyeonhong/nlq/ego4d/CLIP/clip_ego4d_feature')
            self.p_feature = Path(f'/data/soyeonhong/nlq/ego4d/CLIP/__clip_ego4d_feature')
            valid_ids = [p.stem for p in self.p_feature.glob('*.pth')]
            self.annotations = [a for a in self.annotations if a['video_id'] in valid_ids]
        elif self.feature_type == 'egovlp':
            self.p_feature = Path(f'/data/soyeonhong/nlq/nlq_lightning/data/egovlp/{self.caption_type}')
            valid_ids = [p.stem for p in self.p_feature.glob('*.pt')]
            self.annotations = [a for a in self.annotations if a['video_id'] in valid_ids]
        self.max_v_len = 1200

    def __len__(self):
        return len(self.annotations)
    
    def _get_video_feature(self, video_id):
        video_feature = torch.from_numpy(self.video_features[video_id][:])
        v_len = video_feature.shape[0]
        sample_ratio = 1.0
        if v_len > self.max_v_len:
            sample_idx = torch.linspace(0, v_len-1, self.max_v_len).long()
            video_feature = video_feature[sample_idx]
            sample_ratio = self.max_v_len / v_len
            v_len = self.max_v_len
        return video_feature, v_len, sample_ratio

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        
        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)
        
        if self.feature_type == 'clip' or self.feature_type == 'egovlp':
            clip_feature = torch.load(self.p_feature / f"{video_id}.pth")
            
            if not clip_feature.shape[0] == v_len:
                video_feature = interp_t(clip_feature, v_len)
        
        v_mask = torch.ones(v_len)
        
        return {
            'video_id': video_id,
            'v_feat': video_feature,
            'v_mask': v_mask,
        }
        
def generate_mask(mask_size):
    """
    Generates a contrastive kernel (mask) dynamically based on the given size.

    Args:
        mask_size (int): Size of the mask (must be an odd number).
        device (str): Device on which to create the mask ('cpu' or 'cuda').

    Returns:
        torch.Tensor: Generated mask of shape [mask_size, mask_size].
    """
    
    if mask_size % 2 == 0:
        raise ValueError("mask_size must be an odd number.")

    # Initialize the mask with zeros
    mask = torch.zeros((mask_size, mask_size), device='cuda')

    # Define the center and range for positive/negative values
    mid = mask_size // 2

    # Fill positive and negative regions
    for i in range(mid):
        mask[i, :mid + 1] = 1.0  # Top-left positive region
        mask[-(i + 1), :mid + 1] = -1.0  # Bottom-left negative region
        mask[i, -(mid + 1):] = -1.0  # Top-right negative region
        mask[-(i + 1), -(mid + 1):] = 1.0  # Bottom-right positive region

    return mask
        
def generate_scene(src_vid, src_vid_mask, mask):
    # src_vid: [batch_size, L_vid, D_vid]
    # src_vid_mask: [batch_size, L_vid], containing 0 on padded pixels,
    
    bsz, L_src, _ = src_vid.size()

    # temporal self-similarity matrix (TSM)
    norm_vid = src_vid / (src_vid.norm(dim=2, keepdim=True)+1e-8)
    tsm = torch.bmm(norm_vid, norm_vid.transpose(1,2)) # [bsz,L_src,L_src]
    
    # Contrastive kernel
    mask_size = mask.size(0)
    mask = mask.view(1,mask_size,mask_size)
    pad_tsm = nn.ZeroPad2d(mask_size//2)(tsm)
    score = torch.diagonal(F.conv2d(pad_tsm.unsqueeze(1).float(), mask.unsqueeze(1)).squeeze(1), dim1=1,dim2=2)  # [bsz,L_src]
    # average score as threshold
    tau = score.mean(1).unsqueeze(1).repeat(1,L_src)
    # fill the start, end indices with the max score
    L_vid = torch.count_nonzero(src_vid_mask,1)
    st_ed = torch.cat([torch.zeros_like(L_vid).unsqueeze(1), L_vid.unsqueeze(1)-1], dim=-1)
    score[torch.arange(score.size(0)).unsqueeze(1), st_ed] = 100
    # adjacent point removal and thresholding
    score_r = torch.roll(score,1,-1)
    score_l = torch.roll(score,-1,-1)
    # thresholding
    bnds = torch.where((score_r<=score) & (score_l<=score) & (tau<=score), 1., 0.) # [bsz, L_src]
    
    bnd_indices = bnds.nonzero()
    temp = torch.roll(bnd_indices, 1, 0)
    center = (bnd_indices + temp) / 2
    width = bnd_indices - temp

    bnd_spans = torch.cat([center, width[:,1:]], dim=-1)
    bnd_spans = bnd_spans[bnd_spans[:,2] > 0] # remove the width < 0 spans
    
    # Convert [center, width] to [start, end]
    start_end_spans = []
    max_num_events = 0
    for i in range(bsz):
        spans = bnd_spans[bnd_spans[:, 0] == i, 1:]  # Extract spans for the batch
        start_time = spans[:, 0] - spans[:, 1] / 2   # center - width / 2
        end_time = spans[:, 0] + spans[:, 1] / 2     # center + width / 2
        start_end = torch.stack([start_time, end_time], dim=1)  # Combine as [start, end]
        start_end_spans.append(start_end)
        max_num_events = max(max_num_events, start_end.size(0))  # Track max length

    return start_end_spans


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_size', type=int, default=441)
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--dir', type=str, default='/data/soyeonhong/nlq/nlq_lightning/data/scene')
    parser.add_argument('--feature_type', type=str, default='egovlp_internvideo')
    parser.add_argument('--caption_type', type=str, default='videorecap')
    args = parser.parse_args()

    ds = NLQDataset(feature_type=args.feature_type, caption_type=args.caption_type)

    p_out_root_dir = Path(args.dir)
    if args.feature_type == 'clip':
        p_out_dir = p_out_root_dir / args.feature_type
    elif args.feature_type == 'egovlp_internvideo':
        p_out_dir = p_out_root_dir / args.feature_type / f'mask_{args.mask_size}'
    else:
        p_out_dir = p_out_root_dir / args.feature_type / args.caption_type
    p_out_dir.mkdir(parents=True, exist_ok=True)

    cmd = "scontrol show jobid ${SLURM_JOB_ID} | grep -oP '(?<=BatchFlag=)([0-1])'"
    batch_flag = int(os.popen(cmd).read().strip())
    disable = batch_flag == 1
    if not disable:
        print("BatchFlag is 0. tqdm is enabled.")

    mask = generate_mask(args.mask_size).cuda()
    for annotation in tqdm(ds, disable=disable):
        
        video_id = annotation['video_id']
        out_file = p_out_dir / f"{video_id}.pt"
        
        if out_file.exists():
            continue
        video_feature = annotation['v_feat'].cuda()
        video_mask = annotation['v_mask'].cuda()
        scene = generate_scene(video_feature.unsqueeze(0), video_mask.unsqueeze(0), mask)

        scene = scene[0].detach().cpu()
        torch.save(scene, out_file)


if __name__ == '__main__':
    main()
