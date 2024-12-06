import torch
import pytorch_lightning as pl
from hydra.utils import instantiate
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import OneCycleLR

from eval import calc_metrics
from eval_nlq import ReferringRecall

class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.dataset.tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = instantiate(config.model, max_v_len=config.dataset.max_v_len)
        self.nlq_evaluator = ReferringRecall(
            dataset="ego4d",
            ann_dir=config.dataset.ann_dir
        )
        self._log_indices = {}

    def training_step(self, batch, batch_idx):
        torch.cuda.empty_cache()
        total_loss, ce_loss, time_loss = self.model(**batch)
        self.log('total_loss', total_loss, rank_zero_only=True)
        self.log('ce_loss', ce_loss, rank_zero_only=True)
        self.log('time_loss', time_loss, rank_zero_only=True)
        return {
            'loss': total_loss,
        }
    
    def validation_step(self, batch, batch_idx, dataloader_idx): # 0: val, 1: train
        nlq_results, answer_tokens = self.model.generate(**batch)
        pred_answer = self.tokenizer.batch_decode(answer_tokens, skip_special_tokens=True)
        return {
            'question': batch['q_text'],
            'video_id': batch['video_id'],
            'answer': batch['a_text'] if 'a_text' in batch else '',
            'pred_answer': pred_answer,
            'nlq_results': nlq_results,
            'query_id': batch['query_id'],
            'sample_ratio': batch['sample_ratio'],
            'task': batch['task']
        }
    
    def test_step(self, batch, batch_idx, dataloader_idx):
        pass
    
    def predict_step(self, batch, batch_idx, dataloader_idx):
        return self.validation_step(batch, batch_idx, dataloader_idx)
    
    def aggregate_metrics(self, outputs, prefix):
        # evaluate CloseQA
        all_hypos = []
        all_targets = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] == 'CloseQA':
                    all_hypos.append(output['pred_answer'][i])
                    all_targets.append(output['answer'][i])
        if len(all_hypos) > 0:
            num_correct = 0
            for hypo, target in zip(all_hypos, all_targets):
                if hypo == target:
                    num_correct += 1
            acc = num_correct / len(all_targets) * 100
            metrics = {f'{prefix}_close_acc': acc}
        else:
            metrics = {}

        # evaluate OpenQA
        all_hypos = []
        all_targets = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] == 'OpenQA':
                    all_hypos.append(output['pred_answer'][i])
                    all_targets.append(output['answer'][i])
        if len(all_hypos) > 0:
            open_qa_metrics = calc_metrics(all_hypos, [[x] for x in all_targets], test=prefix=='test')
            for k, v in open_qa_metrics.items():
                metrics[f'{prefix}_{k}'] = v

        # evalute NLQ
        nlq_preds = []
        for output in outputs:
            for i in range(len(output['video_id'])):
                if output['task'][i] != 'NLQ':
                    continue
                qid = output['query_id'][i]
                temp_list = qid.split("_")
                sample_ratio = output['sample_ratio'][i]
                new_prediction = [
                    [   segment[0] / sample_ratio,
                        segment[1] / sample_ratio,
                        score  ] 
                    for segment, score in zip(
                        output['nlq_results'][i]['segments'].cpu().detach().tolist(),
                        output['nlq_results'][i]['scores'].cpu().detach().tolist(),
                )]
                nlq_preds.append({
                    'query_idx': int(temp_list[1]),
                    'annotation_uid': temp_list[0],
                    'predicted_times': new_prediction,
                    'clip_uid': output['video_id'][i],
                    'split': prefix
                })
        if len(nlq_preds) > 0:
            performance, score_str = self.nlq_evaluator.evaluate(nlq_preds, verbose=False)
            metrics[f'{prefix}_R1_03'] = performance[0, 0] * 100
            metrics[f'{prefix}_R5_03'] = performance[0, 1] * 100
            metrics[f'{prefix}_R1_05'] = performance[1, 0] * 100
            metrics[f'{prefix}_R5_05'] = performance[1, 1] * 100
            metrics[f'{prefix}_Mean_R1'] = (performance[0, 0] + performance[1, 0]) * 100 / 2

        return metrics

    # def training_epoch_end(self, outputs):
        # self._log_some_outputs(outputs, 'train')
        # metrics = self.aggregate_metrics(outputs, prefix='train')
        # self.log_dict(metrics, sync_dist=True)

    def validation_epoch_end(self, outputs):
        def _mean(key):
            return torch.stack([data[key] for data in outputs]).mean()
                
        val_metrics = self.aggregate_metrics(outputs[0], prefix='val')
        train_metrics = self.aggregate_metrics(outputs[1], prefix='train')
        
        val_metrics.update({
            f'val_{name}': _mean(name) for name in outputs[0][0].keys() if 'loss' in name
        })
        
        train_metrics.update({
            f'train_{name}': _mean(name) for name in outputs[1][0].keys() if 'loss' in name
        })
        
        metrics = {**val_metrics, **train_metrics}
        self.log_dict(metrics, sync_dist=True)

    def configure_optimizers(self):
        optimizer = instantiate(
            self.config.optim.optimizer,
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.config.optim.optimizer.lr
        )
        if self.config.optim.lr_scheduler:
            lr_scheduler = OneCycleLR(
                optimizer=optimizer,
                max_lr=self.config.optim.optimizer.lr,
                total_steps=self.config.total_steps,
                anneal_strategy='linear'
            )
            return {
                'optimizer': optimizer, 
                'lr_scheduler': {
                    'scheduler': lr_scheduler,
                    'interval': 'step'
                }
            }
        else:
            return optimizer