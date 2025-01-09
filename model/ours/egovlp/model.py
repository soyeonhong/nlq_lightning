import os
from abc import abstractmethod

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from .video_transformer import SpaceTimeTransformer


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    def __init__(self, device=None, **kwargs):
        super().__init__(**kwargs)
        if device is not None:
            self = self.to(device)

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum(np.prod(p.size()) for p in model_parameters)
        return super().__str__() + '\nTrainable parameters: {}'.format(params)



def state_dict_data_parallel_fix(load_state_dict, curr_state_dict):
    load_keys = list(load_state_dict.keys())
    curr_keys = list(curr_state_dict.keys())

    redo_dp = False
    undo_dp = False
    if not curr_keys[0].startswith('module.') and load_keys[0].startswith('module.'):   # this
        undo_dp = True
    elif curr_keys[0].startswith('module.') and not load_keys[0].startswith('module.'):
        redo_dp = True

    if undo_dp: # this
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = k[7:]  # remove `module.`
            new_state_dict[name] = v
        # load params
    elif redo_dp:
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in load_state_dict.items():
            name = 'module.' + k  # remove `module.`
            new_state_dict[name] = v
    else:
        new_state_dict = load_state_dict
    return new_state_dict


class FrozenInTime(BaseModel):
    def __init__(self,
                video_params={
                    'model': 'SpaceTimeTransformer',
                    'arch_config': 'base_patch16_224',
                    'pretrained': True,
                    'num_frames': 16,
                    'time_init': 'zeros',
                },
                text_params={
                    'model': 'distilbert-base-uncased',
                    'pretrained': True,
                    'input': 'text',
                },
                projection_dim=256,
                load_checkpoint=None,
                projection='minimal',
                load_temporal_fix='zeros',
                device=None,
                **kwargs):
        super().__init__(**kwargs)

        self.video_params = video_params
        self.text_params = text_params
        self.projection_dim = projection_dim
        self.load_temporal_fix = load_temporal_fix
        if not text_params['pretrained']:
            raise NotImplementedError("Huggingface text models require pretrained init.")

        # pdb.set_trace()
        if self.text_params['model'].startswith('distilbert'):
            self.text_model = AutoModel.from_pretrained('distilbert-base-uncased')
        else:
            self.text_model = AutoModel.from_pretrained(text_params['model'])
        self.text_model.train()

        if video_params['model'] == "SpaceTimeTransformer":
            num_frames = video_params.get('num_frames', 4)
            time_init = video_params.get('time_init', 'zeros')
            attention_style = video_params.get('attention_style', 'frozen-in-time')
            arch_config = video_params.get('arch_config', 'base_patch16_224')
            if arch_config == 'base_patch16_224':
                model = SpaceTimeTransformer(num_frames=num_frames,
                                            time_init=time_init,
                                            attention_style=attention_style)
            else:
                raise NotImplementedError('Only arch_config="base_patch16_224" is supported for now.')

            model.head = nn.Identity()
            model.pre_logits = nn.Identity()
            ftr_dim = model.embed_dim
            if load_checkpoint in ["", None]:
                vit_checkpoint = torch.load(load_checkpoint, map_location="cpu")
                new_vit_dict = state_dict_data_parallel_fix(vit_checkpoint, model.state_dict())
                model.load_state_dict(new_vit_dict, strict=False)
            # for backwards compatibility (old models)
            model.fc = nn.Identity()
            self.video_model = model
        else:
            raise NotImplementedError(f"{video_params['model']} not implemented")

        # Project to a common embedding
        if projection == 'minimal':
            txt_proj = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.text_model.config.hidden_size, projection_dim))
            vid_proj = nn.Sequential(
                nn.Linear(ftr_dim, projection_dim))
        elif projection == '':
            txt_proj = nn.Identity()
            vid_proj = nn.Identity()
        else:
            raise NotImplementedError
        self.txt_proj = txt_proj
        self.vid_proj = vid_proj

        if load_checkpoint not in ["", None]:
            # checkpoint = torch.load(load_checkpoint)
            local_rank = int(os.environ.get('LOCAL_RANK', 0))  # fixed by qinghong.
            checkpoint = torch.load(
                load_checkpoint,
                map_location='cuda:{}'.format(local_rank) if torch.cuda.is_available() else 'cpu')
            state_dict = checkpoint['state_dict']
            new_state_dict = state_dict_data_parallel_fix(state_dict, self.state_dict())
            new_state_dict = self._inflate_positional_embeds(new_state_dict)
            self.load_state_dict(new_state_dict, strict=True)

        if device is not None:
            self.set_device(device)

    def set_device(self, device):
        self.device = device
        self.to(device)

    def forward(self, data, video_only=False, return_embeds=True):
        if video_only:
            video_data = data['video']
            video_embeddings = self.compute_video(video_data)
            return video_embeddings

        text_data = data['text']
        video_data = data['video']

        text_embeddings = self.compute_text(text_data)
        video_embeddings = self.compute_video(video_data)

        if return_embeds:
            return text_embeddings, video_embeddings

        return sim_matrix(text_embeddings, video_embeddings)

    def compute_text(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state[:, 0, :]
        else:
            raise NotImplementedError
        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_text_tokens(self, text_data):
        if self.text_params['model'].startswith('bert'):
            text_embeddings = self.text_model(text_data['input_ids'], attention_mask=text_data['attention_mask'])[
                'pooler_output']    # not implement for bert
        elif self.text_params['model'].startswith('distilbert'):
            text_embeddings = self.text_model(**text_data).last_hidden_state
        else:
            raise NotImplementedError

        text_embeddings = self.txt_proj(text_embeddings)
        return text_embeddings

    def compute_video(self, video_data):
        video_embeddings = self.video_model(video_data)
        video_embeddings = self.vid_proj(video_embeddings)
        return video_embeddings

    def _inflate_positional_embeds(self, new_state_dict):
        # allow loading of timesformer with fewer num_frames
        curr_keys = list(self.state_dict().keys())
        if 'video_model.temporal_embed' in new_state_dict and 'video_model.temporal_embed' in curr_keys:
            load_temporal_embed = new_state_dict['video_model.temporal_embed']
            load_num_frames = load_temporal_embed.shape[1]
            curr_num_frames = self.video_params['num_frames']
            embed_dim = load_temporal_embed.shape[2]

            if load_num_frames != curr_num_frames:
                if load_num_frames > curr_num_frames:
                    print(f'### loaded {self.video_params["model"]} model has MORE frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    new_temporal_embed = load_temporal_embed[:, :curr_num_frames, :]
                else:
                    print(f'### loaded {self.video_params["model"]} model has FEWER frames than current...'
                          f'### loading weights, filling in the extras via {self.load_temporal_fix}')
                    if self.load_temporal_fix == 'zeros':
                        new_temporal_embed = torch.zeros([load_temporal_embed.shape[0], curr_num_frames, embed_dim])
                        new_temporal_embed[:, :load_num_frames] = load_temporal_embed
                    elif self.load_temporal_fix in ['interp', 'bilinear']:
                        # interpolate
                        # unsqueeze so pytorch thinks its an image
                        mode = 'nearest'
                        if self.load_temporal_fix == 'bilinear':
                            mode = 'bilinear'
                        load_temporal_embed = load_temporal_embed.unsqueeze(0)
                        new_temporal_embed = F.interpolate(load_temporal_embed,
                                                           (curr_num_frames, embed_dim), mode=mode, align_corners=True).squeeze(0)
                    else:
                        raise NotImplementedError
                new_state_dict['video_model.temporal_embed'] = new_temporal_embed
        # allow loading with smaller spatial patches. assumes custom border crop, to append the
        # border patches to the input sequence
        if 'video_model.pos_embed' in new_state_dict and 'video_model.pos_embed' in curr_keys:
            load_pos_embed = new_state_dict['video_model.pos_embed']
            load_num_patches = load_pos_embed.shape[1]
            curr_pos_embed = self.state_dict()['video_model.pos_embed']
            if load_num_patches != curr_pos_embed.shape[1]:
                raise NotImplementedError(
                    'Loading models with different spatial resolution / patch number not yet implemented, sorry.')

        return new_state_dict


class TextOnlyFrozenInTime(BaseModel):
    def __init__(self, device=None, ckpt_path=None, **kwargs):
        super().__init__(**kwargs)
        model = FrozenInTime(device=device, load_checkpoint=ckpt_path, **kwargs)
        model.video_model = nn.Identity()
        model.vid_proj = nn.Identity()
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')

    @property
    def device(self):
        return self.model.device
    
    def forward(self, text_data):
        return self.model.compute_text(text_data)


def sim_matrix(a, b, eps=1e-8):
    """
    added eps for numerical stability
    """
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt