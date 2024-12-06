import json
from pathlib import Path
from argparse import Namespace

import hydra
from omegaconf import OmegaConf, DictConfig, open_dict

import pytorch_lightning as pl
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, BasePredictionWriter, ModelSummary
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from eval_nlq import ReferringRecall


class NLQWriter(BasePredictionWriter):
    def __init__(self,
        output_dir,
        official_anns_dir = None,
        test_submit = False,
    ):
        super().__init__(write_interval="batch")
        self.p_outdir = Path(output_dir)
        self.p_int_pred = self.p_outdir / 'intermediate_predictions.json'
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

    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
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
        trainer.accelerator.barrier()

    def write_on_batch_end(self, trainer, pl_module, prediction: list[dict], batch_indices, batch, batch_idx, dataloader_idx):
        split = 'val' if dataloader_idx == 0 else 'train'
        for i in range(len(prediction['video_id'])):
            qid = prediction['query_id'][i]
            temp_list = qid.split("_")
            sample_ratio = prediction['sample_ratio'][i]
            new_prediction = [
                [   segment[0] / sample_ratio,
                    segment[1] / sample_ratio] 
                for segment in prediction['nlq_results'][i]['segments'].cpu().detach().tolist()]
            
            result = {
                'question': prediction['question'][i],
                'query_idx': int(temp_list[1]),
                'annotation_uid': temp_list[0],
                'predicted_times': new_prediction,
                'clip_uid': prediction['video_id'][i],
                'split': split
            }
            
            self.rank_seg_preds[split].append(result)
            
        with open(self.p_tmp_outdir / f'rank-{trainer.global_rank}-{split}.json', 'w') as f:
            json.dump(self.rank_seg_preds[split], f)

    def on_predict_epoch_end(self, trainer, pl_module, outputs):

        for key in ['train', 'val']:            
            # Write sorted predictions to a JSON file
            with open(self.p_tmp_outdir / f'rank-{trainer.global_rank}-{key}.json', 'w') as f:
                json.dump(self.rank_seg_preds[key], f)

            
        if trainer.world_size > 1:
            trainer.accelerator.barrier()

        if trainer.is_global_zero:
            # get segmented predictions
            print('Gathering predicted timestamps...')
            all_seg_preds_train = []
            all_seg_preds_val = []
            for p_pt in self.p_tmp_outdir.glob('*.json'):
                rank_seg_preds = json.loads(p_pt.read_text())
                if 'val' in p_pt.name:
                    all_seg_preds_val.extend(rank_seg_preds)
                else:
                    all_seg_preds_train.extend(rank_seg_preds)

            for key in ['train', 'val']:
                p_pred = self.p_outdir / f'{key}_predictions.json'
                with open(p_pred, 'w') as f:
                    json.dump(all_seg_preds_val if key == 'val' else all_seg_preds_train, f)

            # TODO: test_submit
            if self.test_submit:
                print("test_submit")

            if not self.test_submit:
                for idx, result in enumerate([all_seg_preds_val, all_seg_preds_train]):
                    
                    split = 'Validation' if idx == 0 else 'Train'
                    
                    performance, score_str = self.nlq_evaluator.evaluate(result, verbose=False)

                    recall = self.nlq_evaluator.display_results(performance, f'{split} performance')
                    
                    print(recall, flush=True)

            # remove temporary files
            for p_tmp in self.p_tmp_outdir.glob('*'):
                p_tmp.unlink()
            self.p_tmp_outdir.rmdir()
            
def _adjust_ddp_config(trainer_cfg):
    trainer_cfg = dict(trainer_cfg)
    strategy = trainer_cfg.get('strategy', None)
    if strategy is None:
        trainer_cfg['strategy'] = DDPPlugin(
            find_unused_parameters=trainer_cfg['find_unused_parameters'], 
            gradient_as_bucket_view=True)
    return trainer_cfg

def get_trainer(config, jid, enable_progress_bar=False, enable_checkpointing=True, ddp_timeout=30):
    runtime_outdir: str = config.runtime_outdir
    trainer_config = Namespace(**_adjust_ddp_config(config.trainer))
    loggers_config = config.trainer.logger

    # Callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        LearningRateMonitor(logging_interval='step'),
        TQDMProgressBar(refresh_rate=1 if enable_progress_bar else 20),
        NLQWriter(output_dir=runtime_outdir,
                  official_anns_dir=config.dataset.ann_dir)
        ]

    # Add Model Checkpoints based on test_splits
    if enable_checkpointing:
        if 'NLQ_val' in config.dataset.test_splits:
            ckpt_callback_r103 = ModelCheckpoint(
                dirpath=runtime_outdir,
                save_last=False,
                monitor='val_R1_03',
                mode='max',
                save_top_k=1,
                filename='{epoch}-val_R1_03={val_R1_03:.3f}')
            ckpt_callback_r503 = ModelCheckpoint(
                dirpath=runtime_outdir,
                save_last=False,
                monitor='val_R5_03',
                mode='max',
                save_top_k=1,
                filename='{epoch}-val_R5_03={val_R5_03:.3f}')
            callbacks.extend([ckpt_callback_r103, ckpt_callback_r503])
    else:
        ckpt_callback_r103 = None
        ckpt_callback_r503 = None
            
    loggers = [CSVLogger(save_dir=runtime_outdir, name="lit", version=jid)]
    for logger_config in loggers_config:
        logger: WandbLogger = hydra.utils.instantiate(logger_config)
        loggers.append(logger)

    trainer = pl.Trainer.from_argparse_args(
        trainer_config,
        logger=loggers,
        callbacks=callbacks,
        accelerator="gpu")
    
    return trainer, ckpt_callback_r503
