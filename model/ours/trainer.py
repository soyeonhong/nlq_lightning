import json
from pathlib import Path
from argparse import Namespace

import hydra
import torch
from omegaconf import OmegaConf, DictConfig, open_dict

# This is old version
# import pytorch_lightning as pl
# from pytorch_lightning.plugins import DDPPlugin
# from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, BasePredictionWriter, ModelSummary
# from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger, CSVLogger
# from eval_nlq import ReferringRecall

import lightning as L
from lightning.pytorch.strategies import DDPStrategy
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, BasePredictionWriter, ModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger, CSVLogger
from eval_nlq import ReferringRecall

type_loggers = WandbLogger | CSVLogger

class NLQWriter(BasePredictionWriter):
    def __init__(self,
        output_dir,
        official_anns_dir = None,
        test_submit = False,
    ):
        super().__init__(write_interval="batch")
        self.p_outdir = Path(output_dir)
        self.p_int_pred = self.p_outdir / 'intermediate_predictions.json'
        self.p_pred = self.p_outdir / 'predictions.json'
        self.p_metrics = self.p_outdir / 'metrics.json'
        self.p_metrics_log = self.p_outdir / 'metrics.log'
        self.rank_seg_preds = {
            'train': [],
            'val': []
        }
        self.test_submit = test_submit
        self.official_anns_dir = Path(official_anns_dir)
        
        self.nlq_evaluator = ReferringRecall(
            dataset="ego4d",
            ann_dir=self.official_anns_dir
        )

        if self.test_submit:
            self.split = 'test_unannotated'
            self.p_pred = self.p_pred.with_name('test_predictions.json')
            self.p_int_pred = self.p_int_pred.with_name('test_intermediate_predictions.pt')
        else:
            self.split = 'val'

        self.p_tmp_outdir = self.p_outdir / 'tmp' / self.split

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str):
        if trainer.is_global_zero:
            print(f'Begin setup... for stage {stage}')
            print(f'Output directory: {self.p_outdir}')
            if self.p_tmp_outdir.exists():
                print(f'Removing existing temporary files in {self.p_tmp_outdir}...')
                for p_tmp in self.p_tmp_outdir.glob('*'):
                    p_tmp.unlink()
            else:
                print(f'Creating temporary directory... {self.p_tmp_outdir}')
                self.p_tmp_outdir.mkdir(parents=True, exist_ok=True)
            print('Setup done.')
        trainer.strategy.barrier()

    def write_on_batch_end(self, trainer, pl_module, prediction: list[dict], batch_indices, batch, batch_idx, dataloader_idx):
        # prediction key: 'qeustion', 'video_id', 'answer', 'pred_answer', 'nlq_results', 'query_id', 'sample_ratio', 'task'

        split = 'val' if dataloader_idx == 0 else 'train'
        for i in range(len(prediction['video_id'])):
            qid = prediction['query_id'][i]
            temp_list = qid.split("_")
            sample_ratio = prediction['sample_ratio'][i]
            new_prediction = [
                [   segment[0] / sample_ratio,
                    segment[1] / sample_ratio,
                    score  ] 
                for segment, score in zip(
                    prediction['nlq_results'][i]['segments'].cpu().detach().tolist(),
                    prediction['nlq_results'][i]['scores'].cpu().detach().tolist(),
            )]
            
            result = {
                'query_idx': int(temp_list[1]),
                'annotation_uid': temp_list[0],
                'predicted_times': new_prediction,
                'clip_uid': prediction['video_id'][i],
                'split': split
            }
            
            self.rank_seg_preds[split].append(result)

        if batch_idx % 100 == 0:  # checkpointing
            self.rank_seg_preds[split] = sorted(self.rank_seg_preds[split], key=lambda x: x[:-1])
            
            with open(self.p_tmp_outdir / f'rank-{trainer.global_rank}-{split}.json', 'w') as f:
                json.dump(self.rank_seg_preds[split], f)

    def on_predict_epoch_end(self, trainer, pl_module):

        for key in ['train', 'val']:
            # Sort the predictions based on elements excluding the last one
            self.rank_seg_preds[key].sort(key=lambda x: x[:-1])
            
            # Write sorted predictions to a JSON file
            with open(self.p_tmp_outdir / f'rank-{trainer.global_rank}-{key}.json', 'w') as f:
                json.dump(self.rank_seg_preds[key], f)

            
        if trainer.world_size > 1:
            trainer.strategy.barrier()

        if trainer.is_global_zero:
            # get segmented predictions
            print('Gathering segmented features...')
            all_seg_preds_train = []
            all_seg_preds_val = []
            for p_pt in self.p_tmp_outdir.glob('*.json'):
                rank_seg_preds = json.loads(p_pt.read_text())
                if 'val' in p_pt.name:
                    all_seg_preds_val.extend(rank_seg_preds)
                else:
                    all_seg_preds_train.extend(rank_seg_preds)
            
            # TODO: test_submit
            if self.test_submit:
                print("test_submit")

            # write the final predictions to json
            # train, val
            # json.dump(all_seg_preds, self.p_pred.open('w'))

            if not self.test_submit:
                metrics = {}
                for idx, result in enumerate([all_seg_preds_val, all_seg_preds_train]):
                    
                    split = 'val' if idx == 0 else 'train'
                    
                    performance, score_str = self.nlq_evaluator.evaluate(result, verbose=False)
                    metrics[f'{split}_R1_03'] = performance[0, 0] * 100
                    metrics[f'{split}_R5_03'] = performance[0, 1] * 100
                    metrics[f'{split}_R1_05'] = performance[1, 0] * 100
                    metrics[f'{split}_R5_05'] = performance[1, 1] * 100
                    metrics[f'{split}_Mean_R1'] = (performance[0, 0] + performance[1, 0]) * 100 / 2

            # remove temporary files
            for p_tmp in self.p_tmp_outdir.glob('*'):
                p_tmp.unlink()
            self.p_tmp_outdir.rmdir()
            
# def _adjust_ddp_config(trainer_cfg):
#     trainer_cfg = dict(trainer_cfg)
#     strategy = trainer_cfg.get('strategy', None)
#     if strategy is None:
#         trainer_cfg['strategy'] = DDPPlugin(
#             find_unused_parameters=trainer_cfg['find_unused_parameters'], 
#             gradient_as_bucket_view=True)
#     return trainer_cfg

def get_trainer(config, jid, enable_progress_bar=False, enable_checkpointing=True, ddp_timeout=30):
    runtime_outdir: str = config.runtime_outdir
    trainer_config: DictConfig = config.trainer
    # trainer_config = Namespace(**_adjust_ddp_config(config.trainer))

    # Callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=1 if enable_progress_bar else 20),
        NLQWriter(output_dir=runtime_outdir,
                  official_anns_dir=config.dataset.ann_dir)
    ]

    # Add Model Checkpoints based on test_splits
    checkpoint_callbacks = []
    if 'QaEgo4D_test' in config.dataset.test_splits:
        checkpoint_callbacks.append(
            ModelCheckpoint(
                dirpath=runtime_outdir,
                save_last=False,
                monitor='val_ROUGE',
                mode='max',
                save_top_k=1,
                filename='{step}-val_ROUGE={val_ROUGE:.3f}'
            )
        )
    if 'QaEgo4D_test_close' in config.dataset.test_splits:
        checkpoint_callbacks.append(
            ModelCheckpoint(
                dirpath=runtime_outdir,
                save_last=False,
                monitor='val_close_acc',
                mode='max',
                save_top_k=1,
                filename='{step}-val_close_acc={val_close_acc:.3f}'
            )
        )
    if 'NLQ_val' in config.dataset.test_splits:
        checkpoint_callbacks.append(
            ModelCheckpoint(
                dirpath=runtime_outdir,
                save_last=False,
                monitor='val_R1_03',
                mode='max',
                save_top_k=1,
                filename='{epoch}-val_R1_03={val_R1_03:.3f}'
            )
        )

    if enable_checkpointing:
        callbacks.extend(checkpoint_callbacks)

    # Logger
    # logger = TensorBoardLogger(
    #     save_dir=runtime_outdir,
    #     version=jid,
    #     name="lit",
    #     default_hp_metric=False
    # )
    
    assert jid is not None, 'jid must be provided when loggers are enabled'
    with open_dict(trainer_config):  # obtaining write access
        loggers_config = trainer_config.pop('logger', [])  # to not pass it to the Trainer
        
    loggers = [CSVLogger(save_dir=runtime_outdir, name="lit", version=jid)]
    for logger_config in loggers_config:
        logger: type_loggers = hydra.utils.instantiate(logger_config)
        loggers.append(logger)
    
    trainer_config = OmegaConf.to_container(trainer_config, resolve=True)
    if 'strategy' not in trainer_config:
        trainer_config['strategy'] = DDPStrategy(
            find_unused_parameters=True)

    # trainer = L.Trainer.from_argparse_args(
    #     trainer_config,
    #     logger=logger,
    #     callbacks=callbacks)
    
    trainer = L.Trainer(
        **trainer_config,
        enable_model_summary=False,
        default_root_dir=runtime_outdir,
        logger=logger,
        callbacks=callbacks,
    )

    return trainer, checkpoint_callbacks
