# Joint dataset of CloseQA, OpenQA, and NLQ

import os
import math
import json
import random
from pathlib import Path
from typing import Iterable
from einops import rearrange
from omegaconf import OmegaConf

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer
from pytorch_lightning.utilities.distributed import rank_zero_only

templates = [
    "What color is {object}?", # "Objects: What X is Y?"
    "In what location did I see {object}?", # "Objects: In what location did I see object X ?"
    "Where is {object}?", # "Objects: Where is object X?", "Objects: Where is object X?"
    "How many {object}?" #  "Objects: How many X’s? (quantity question)"
]

@rank_zero_only
def log_data_info(train_dataset, val_dataset, test_dataset):
    print(f'#train: {len(train_dataset)}')
    print(f'#val: {len(val_dataset)}')
    print(f'#test: {len(test_dataset)}')


class BaseDataset(Dataset):
    def __init__(self, config, data_dir, split, feature_type, max_v_len):
        super().__init__()
        self.split = split
        self.video_features = h5py.File(os.path.join(data_dir, feature_type + '.hdf5'), 'r')
        self.object_qa = config.get('object_qa', None)
        self.object_aug = config.get('object_aug', None)
        self.aug_from_query = config.get('from_query', None)
        self.jitter = config.get('jitter', None)
        self.search_all = config.get('search_all', None)
        self.nlq_from_qa = config.get('nlq_from_qa', None)
        self.temporal_object_qa = config.get('temporal_object_qa', None)
        
        if (self.object_qa or self.object_aug) and 'train' in self.split:
            self.annotations = json.loads(Path(os.path.join(data_dir, f'annotations.{split}_object.json')).read_text())
            self.p_env_dir = Path(config.env_dir)
            self.env_interval = config.env_interval
            
            if self.object_qa:
                self.slice_int = config.slice_interval
                self.search_int = config.search_interval
            elif self.object_aug:
                self.obj_number = config.obj_number
            
            required_clip_uids = set(a['video_id'] for a in self.annotations)
            valid_clip_uids = set(video_id.stem for video_id in list(self.p_env_dir.glob('**/*.json')))
            diff = required_clip_uids - valid_clip_uids
            print(f'Clips not existing in LLaVA: {diff} ({len(diff)})')
            self.annotations = [a for a in self.annotations if a['video_id'] in valid_clip_uids]
        else:
            self.annotations = json.loads(Path(os.path.join(data_dir, f'annotations.{split}.json')).read_text())
        self.max_v_len = max_v_len
        print(f'{split} set: {len(self.annotations)}')
    
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


class NLQDataset(BaseDataset):
    def __init__(self, config, data_dir, split, feature_type, max_v_len):
        super().__init__(config, data_dir, split, feature_type, max_v_len)

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        query_id = self.annotations[index].get('sample_id')
        question = self.annotations[index]['question']
        clip_duration = self.annotations[index].get('clip_duration')
        if (self.object_qa or self.object_aug) and 'train' in self.split:
            p_env_data = self.p_env_dir / f'{video_id}.json'
        
        if (self.object_qa or self.object_aug) and 'train' in self.split:
            env_datas = json.loads(p_env_data.read_text())
            if self.object_qa:
                ori_video_feature = torch.from_numpy(self.video_features[video_id][:])

        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)

        if 'clip_start_sec' in self.annotations[index]:
            start_time = self.annotations[index].get('clip_start_sec')
            end_time = self.annotations[index].get('clip_end_sec')
        else:
            start_time = self.annotations[index].get('moment_start_frame') / 30
            end_time = self.annotations[index].get('moment_end_frame') / 30

        query_type = self.annotations[index].get('query_type')
        if query_type == 'narration':
            duration = end_time - start_time
            center = (end_time + start_time) / 2
            scale_ratio = random.randint(1, 10)
            shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
            new_center = center - shift_number
            start_time = new_center - scale_ratio * duration / 2
            end_time = new_center + scale_ratio * duration / 2

        segments = torch.tensor([[start_time, end_time]]) * 30 / 16.043 * sample_ratio
        labels = torch.zeros(len(segments), dtype=torch.int64)
        one_hot_labels = F.one_hot(labels, 1)  # (1, 1)
        
        sample = {
            'video_id': video_id,
            'question': f"question: {question} video: ",
            'answer': 'None',
            'v_feat': video_feature,
            'v_len': v_len,
            'segments': segments,
            'one_hot_labels': one_hot_labels,
            'query_id': query_id,
            'sample_ratio': sample_ratio,
            'task': 'NLQ'
        }
        
        if (self.object_qa or self.object_aug) and 'train' in self.split:
            query_objs = self.annotations[index]['entities']['values']
            
            if self.object_qa:
                # positive question
                if len(query_objs) == 1:
                    if query_objs[0] == '':
                        obj_q_pos = question
                    else:
                        obj_q_pos = f"Is there a {query_objs[0]}?"
                else:
                    obj_q_pos = f"Are there {', '.join(query_objs[:-1])} and {query_objs[-1]}?"
                if self.temporal_object_qa:
                    obj_q_pos = "In _ , " + obj_q_pos
                q_str_pos = f"question: {obj_q_pos} video: "
                
                # negative question
                # slice time
                if self.search_all:
                    env_search_idx_s = 0
                    env_search_idx_e = len(env_datas) - 1
                else:
                    time_per_v_feat = clip_duration  / v_len
                    center = (end_time + start_time) // 2
                    slice_t_s = max(0, center - self.slice_int / 2)
                    slice_t_e = min(clip_duration, center + self.slice_int / 2)
                    slice_int_idx = self.slice_int // time_per_v_feat
                    
                    if slice_t_s == 0:
                        slice_idx_s = 0
                        slice_idx_e = slice_int_idx
                    elif slice_t_e == clip_duration:
                        slice_idx_e = v_len - 1
                        slice_idx_s = slice_idx_e - slice_int_idx
                    else:
                        slice_idx_s = slice_t_s // time_per_v_feat
                        slice_idx_e = slice_idx_s + slice_int_idx
                            
                    slice_idx_s , slice_idx_e = int(slice_idx_s), int(slice_idx_e)
                    
                    env_search_idx_s = max(0, math.floor(slice_t_s // self.env_interval) - 1 - self.search_int // 2)
                    env_search_idx_e = min(math.ceil(slice_t_e / self.env_interval) - 1 + self.search_int // 2, len(env_datas) - 1)
                
                # select negative object
                obj_list = []
                obj_gt_list = []
                
                if self.search_all:
                    for env_data in env_datas[env_search_idx_s:env_search_idx_e + 1]:
                        obj_list.extend(env_data['entities']['values'])
                        if start_time <= env_data['start'] <= end_time:
                            obj_gt_list.extend(env_data['entities']['values'])
                else:
                    for env_data in env_datas[env_search_idx_s:env_search_idx_e + 1]:
                        obj_list.extend(env_data['entities']['values'])
                        if slice_t_s <= env_data['start'] <= slice_t_e:
                            obj_gt_list.extend(env_data['entities']['values'])
                        
                obj_list = list(set(obj_list) - set(obj_gt_list))
                
                if obj_list == []:
                    q_str_neg = q_str_pos
                else:      
                    obj_q_neg = f"Is there a {random.choice(obj_list)}?"
                    if self.temporal_object_qa:
                        obj_q_neg = "In _ , " + obj_q_neg
                    q_str_neg = f"question: {obj_q_neg} video: "
                
                sample['question_obj_pos'] = q_str_pos
                sample['question_obj_neg'] = q_str_neg
                sample['answer_obj_pos'] = 'Yes'
                sample['answer_obj_neg'] = 'No'
                if not self.nlq_from_qa:
                    sample['slice'] = [slice_idx_s, slice_idx_e]
                    sample['v_feat_for_obj'] = ori_video_feature[slice_idx_s:slice_idx_e + 1]
                
            if self.object_aug:
                q_aug_strs = []
                if self.aug_from_query:
                    if query_objs[0] == '':
                        q_aug_strs = [question for idx in range(len(templates))]
                    else:
                        for template in templates:
                            q_str = f"question: {template.format(object=query_objs[0])} video: "
                            q_aug_strs.append(q_str)
                    # jitter
                    s_t_jit = max(0, start_time - self.jitter)
                    e_t_jit = min(clip_duration, end_time + self.jitter)
                    seg_jit = torch.tensor([[s_t_jit, e_t_jit]]) * 30 / 16.043 * sample_ratio
                    labels_jit = torch.zeros(len(seg_jit), dtype=torch.int64)
                    one_hot_labels_jit = F.one_hot(labels_jit, 1)
                    
                    sample['segments_jit'] = seg_jit
                    sample['one_hot_labels_jit'] = one_hot_labels_jit

                else:
                    obj_list = []
                    for env_data in env_datas:
                        if start_time <= env_data['start'] <= end_time:
                            obj_list.extend(env_data['entities']['values'])
                    obj_list = list(set(obj_list))
                    
                    if obj_list == [] or len(obj_list) < self.obj_number:
                        aug_objs = obj_list
                        if obj_list == []:
                            for idx in range(self.obj_number):
                                aug_objs.append(question)
                        else:
                            for idx in range(self.obj_number - len(obj_list)):
                                aug_objs.append(random.choice(obj_list))
                    else:
                        aug_objs = random.sample(obj_list, self.obj_number)
                    
                    if obj_list == []:            
                        q_aug_strs = aug_objs
                    else:
                        q_aug_strs = []
                        for obj in aug_objs:
                            for template in templates:
                                q_str = f"question: {template.format(object=obj)} video: "
                                q_aug_strs.append(q_str)
                                
                sample['question_obj_aug'] = q_aug_strs

        return sample

class JointDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[Dataset], tokenizer_path) -> None:
        super().__init__(datasets)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, 
                                                       local_files_only=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token  # BUG: Set this per convenience for GPT-2
        self.object_qa = datasets[0].object_qa
        self.split = datasets[0].split
        self.object_aug = datasets[0].object_aug
        self.aug_from_query = datasets[0].aug_from_query
        self.search_all = datasets[0].search_all

    def collate_fn(self, batch):
        question = [b['question'] for b in batch]
        question_tok = self.tokenizer(question, padding=True, return_tensors='pt', add_special_tokens=False)
        
        answer = [b['answer'] for b in batch]
        labels = self.tokenizer(answer, padding=True, return_tensors='pt').input_ids
        # NOTE: NLQ data does not have an answer
        for idx, a in enumerate(answer):
            if a == 'None':
                labels[idx] = torch.ones_like(labels[idx]) * -100

        video_feature = [b['v_feat'] for b in batch]
        video_feature_padded = pad_sequence(video_feature, batch_first=True)
        video_mask = pad_sequence([torch.ones(len(v)) for v in video_feature], batch_first=True).bool()

        result = {
            'video_id': [b['video_id'] for b in batch],
            'q_text': question,
            'q_token': question_tok.input_ids,
            'q_mask': question_tok.attention_mask.bool(),
            'v_feat': video_feature_padded,
            'v_mask': video_mask,
            'v_len': np.asarray([b['v_len'] for b in batch], dtype=np.long),
            'gt_segments': torch.stack([b['segments'] for b in batch]),
            'gt_labels': torch.stack([b['one_hot_labels'] for b in batch]),
            'query_id': [b['query_id'] for b in batch],
            'sample_ratio': [b['sample_ratio'] for b in batch],
            'a_text': answer,
            'labels': labels,
            'task': [b['task'] for b in batch]
        }
        
        if self.object_qa and 'train' in self.split:
            question_obj_pos = [b['question_obj_pos'] for b in batch]
            question_obj_neg = [b['question_obj_neg'] for b in batch]
            question_obj_pos_tok = self.tokenizer(question_obj_pos, padding=True, return_tensors='pt', add_special_tokens=False)
            question_obj_neg_tok = self.tokenizer(question_obj_neg, padding=True, return_tensors='pt', add_special_tokens=False)
            
            answer_obj_pos = [b['answer_obj_pos'] for b in batch]
            answer_obj_neg = [b['answer_obj_neg'] for b in batch]
            
            labels_obj_pos = self.tokenizer(answer_obj_pos, padding=True, return_tensors='pt').input_ids
            labels_obj_neg = self.tokenizer(answer_obj_neg, padding=True, return_tensors='pt').input_ids
            
            result['q_text_obj_pos'] = question_obj_pos
            result['q_token_obj_pos'] = question_obj_pos_tok.input_ids
            result['q_mask_obj_pos'] = question_obj_pos_tok.attention_mask.bool()
            result['q_text_obj_neg'] = question_obj_neg
            result['q_token_obj_neg'] = question_obj_neg_tok.input_ids
            result['q_mask_obj_neg'] = question_obj_neg_tok.attention_mask.bool()
            result['labels_obj_pos'] = labels_obj_pos
            result['labels_obj_neg'] = labels_obj_neg
            
            if not self.search_all:
                result['slice'] = torch.stack([torch.tensor(b['slice']) for b in batch])
                
                video_feature_for_obj = [b['v_feat_for_obj'] for b in batch]
                video_feature_padded_for_obj = pad_sequence(video_feature_for_obj, batch_first=True)
                video_mask_for_obj = pad_sequence([torch.ones(len(v)) for v in video_feature_for_obj], batch_first=True).bool()
                
                result['v_feat_for_obj'] = video_feature_padded_for_obj
                result['v_mask_for_obj'] = video_mask_for_obj
        
        if self.object_aug and 'train' in self.split:
            B = len(batch)
            num_sen = len(batch[0]['question_obj_aug'])
            question_obj_aug = []
            
            for b in batch:
                question_obj_aug.extend(b['question_obj_aug']) # [B * num_sen]
            question_obj_aug_tok = self.tokenizer(question_obj_aug, padding=True, return_tensors='pt', add_special_tokens=False) # [B * num_sen,  L]
            
            input_ids = question_obj_aug_tok.input_ids # [B * num_sen,  L]
            input_ids = rearrange(input_ids, '(B num_sen) L -> B num_sen L', B=B, num_sen=num_sen) # [B, num_sen, L]
            attention_mask = question_obj_aug_tok.attention_mask.bool() # [B * num_sen,  L]
            attention_mask = rearrange(attention_mask, '(B num_sen) L -> B num_sen L', B=B, num_sen=num_sen) # [B, num_sen, L]
            result['q_text_obj_aug'] = question_obj_aug
            result['q_token_obj_aug'] = input_ids # [B, num_sen, L]
            result['q_mask_obj_aug'] = attention_mask # [B, num_sen, L]
            
            if self.aug_from_query:
                result['gt_segments_jit'] = torch.stack([b['segments_jit'] for b in batch])
                result['gt_labels_jit'] = torch.stack([b['one_hot_labels_jit'] for b in batch])

        return result
    
class QADataset(BaseDataset):
    def __init__(self, data_dir, split, feature_type, max_v_len, qa_type, CloseQA_weight=50):
        super().__init__(data_dir, split, feature_type, max_v_len)
        self.qa_type = qa_type  # CloseQA, OpenQA, Mixed
        self.choice_indices = ['A', 'B', 'C', 'D']
        self.CloseQA_weight = CloseQA_weight
        self.openqa_weight = 100 - CloseQA_weight

    def __getitem__(self, index):
        video_id = self.annotations[index]['video_id']
        query_id = self.annotations[index].get('sample_id')
        question = self.annotations[index]['question']
        answer = self.annotations[index]['answer'].strip()

        qa_type = self.qa_type
        if qa_type == 'Mixed':  # randomly choose a qa type
            qa_type = random.choices(['CloseQA', 'OpenQA'], weights=[self.CloseQA_weight, self.openqa_weight], k=1)[0]
        if qa_type == 'OpenQA':
            question_str = f"question: {question} video: "
            answer_str = answer
        elif qa_type == 'CloseQA':
            wrong_answers = self.annotations[index]['wrong_answers']
            # shuffle choices
            choices = [answer] + wrong_answers
            random.shuffle(choices)
            answer_index = choices.index(answer)
            choices = [f'({self.choice_indices[idx]}) {choices[idx]}' for idx in range(len(choices))]  # ["(A) xx", "(B) xx", "(C) xx", "(D) xx"]
            choices_str = ' '.join(choices)  # (A) xx (B) xx (C) xx (D) xx
            question_str = f"question: {question} choices: {choices_str}. video: "
            answer_str = choices[answer_index]  # (A/B/C/D) xx
        else:
            raise NotImplementedError
        
        video_feature, v_len, sample_ratio = self._get_video_feature(video_id)

        start_frame = self.annotations[index].get('moment_start_frame')
        end_frame = self.annotations[index].get('moment_end_frame')
        start_time = start_frame / 30
        end_time = end_frame / 30

        if 'video_start_sec' not in self.annotations[index]:  # LLM generated QA
            duration = end_time - start_time
            center = (end_time + start_time) / 2
            scale_ratio = random.randint(1, 10)
            shift_number = random.uniform(-1, 1) * (scale_ratio - 1) * duration / 2
            new_center = center - shift_number
            start_time = new_center - scale_ratio * duration / 2
            end_time = new_center + scale_ratio * duration / 2

        segments = torch.tensor([[start_time, end_time]]) * 30 / 16.043 * sample_ratio
        labels = torch.zeros(len(segments), dtype=torch.int64)
        one_hot_labels = F.one_hot(labels, 1)  # (1, 1)

        return {
            'video_id': video_id,
            'question': question_str,
            'answer': answer_str,
            'v_feat': video_feature,
            'v_len': v_len,
            'segments': segments,
            'one_hot_labels': one_hot_labels,
            'query_id': query_id,
            'sample_ratio': sample_ratio,
            'task': qa_type
        }


class JointDataModule(pl.LightningDataModule):
    train_dataset = None
    val_dataset = None
    test_dataset = None

    def __init__(self, config):
        super().__init__()
        self.config = config.dataset
        
        self.save_hyperparameters(OmegaConf.to_container(config, resolve=True))
        
    def setup(self, stage=None):
        CloseQA_weight = self.config.get('closeqa_weight', 50)
        self.train_dataset = JointDataset(                         
            [
                NLQDataset(self.config, self.config.data_dir, train_split, self.config.feature_type, self.config.max_v_len)
                for train_split in self.config.nlq_train_splits
            ],
            self.config.tokenizer_path
        )

        test_datasets = []
        for split in self.config.test_splits:
            if split == 'QaEgo4D_test':
                test_datasets.append(QADataset(self.config, self.config.data_dir, split, self.config.feature_type, self.config.max_v_len, 'OpenQA'))
            elif split == 'QaEgo4D_test_close':
                test_datasets.append(QADataset(self.config, self.config.data_dir, split, self.config.feature_type, self.config.max_v_len, 'CloseQA'))
            elif split in ['NLQ_val', 'NLQ_test_unannotated']:
                test_datasets.append(NLQDataset(self.config, self.config.data_dir, split, self.config.feature_type, self.config.max_v_len))
            else:
                print(split)
                raise NotImplementedError
        self.val_dataset = self.test_dataset = JointDataset(test_datasets, self.config.tokenizer_path)
        
        log_data_info(self.train_dataset, self.val_dataset, self.test_dataset)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=self.config.num_workers,
            collate_fn=self.train_dataset.collate_fn,
            pin_memory=True,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.config.num_workers,
            collate_fn=self.val_dataset.collate_fn,
            pin_memory=True
        )
