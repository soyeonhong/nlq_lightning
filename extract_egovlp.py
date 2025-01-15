import os
import json
import argparse
from pathlib import Path

import h5py
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model.ours.egovlp.model import TextOnlyFrozenInTime

class NLQDataset(Dataset):
    def __init__(self, caption_dir, tokenizer, caption_type):
        self.caption_dir = caption_dir
        self.tokenizer = tokenizer
        self.caption_lists = list(caption_dir.glob('*.json'))
        self.caption_type = caption_type

    def __len__(self):
        return len(self.caption_lists)

    def __getitem__(self, idx):
        p_caption = self.caption_lists[idx]
        clip_uid = p_caption.stem
        
        if self.caption_type == 'videorecap':
            captions = json.loads(p_caption.read_text())['captions']['text']
        elif self.caption_type == 'llava':
            captions = json.loads(p_caption.read_text())['answers']
            captions = [c[-1] for c in captions]
        tokens = self.tokenizer(
            captions, return_tensors='pt', padding=True, truncation=True, max_length=512)

        return {
            'clip_uid': clip_uid,
            'tokens': tokens
        }
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption_type', type=str, default='videorecap')
    args = parser.parse_args()

    if args.caption_type == 'videorecap':
        p_cap = Path('/data/soyeonhong/nlq/nlq_lightning/videorecap_outputs/caption_1s')
    elif args.caption_type == 'llava':
        p_cap = Path('/data/soyeonhong/nlq/nlq_lightning/LLaVA-NeXT-Video-7B-DPO/global_v2')
        
    model = TextOnlyFrozenInTime(ckpt_path="/data/soyeonhong/nlq/nlq_lightning/data/egovlp-config-removed.pth").cuda().eval()
    tokenizer = model.tokenizer
    
    p_out_root_dir = Path(f"/data/soyeonhong/nlq/nlq_lightning/data/egovlp")
    p_out_dir = p_out_root_dir / args.caption_type
    p_out_dir.mkdir(parents=True, exist_ok=True)
    
    ds = NLQDataset(p_cap, tokenizer, args.caption_type)
    
    for annotation in tqdm(ds):
        clip_uid = annotation['clip_uid']
        tokens = annotation['tokens'].to('cuda')
        p_out = p_out_dir / f"{clip_uid}.pt"
        
        if p_out.exists():
            continue
        
        with torch.no_grad():
            embeddings = model(tokens).cpu().detach()
            torch.save(embeddings, p_out)

if __name__ == '__main__':
    main()       