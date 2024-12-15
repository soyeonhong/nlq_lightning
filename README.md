# Our NLQ Method

## Quick Start

### Environment Setup

```bash
# conda (recommended)
conda create -n groundvqa python=3.9 -y && conda activate groundvqa
pip install --upgrade "pip<24.1"
pip install -r requirements.txt
```
- Compile `nms_1d_cpu` following [here](https://github.com/happyharrycn/actionformer_release/blob/main/INSTALL.md)
- Download the data, video feature, and model checkpoints from [Huggingface](https://huggingface.co/Becomebright/GroundVQA)
  - **data:** unzip `data.zip` under the project's root directory.
  - **video feature:** merge the files `cat egovlp_internvideoa* > egovlp_internvideo.hdf5` and put it under `data/unified/`
  - **model checkpoints**: put them under `checkpoints/`


## Training

```bash
bash scripts/train.sbatch
```

## Evaluation

```bash
bash scripts/eval.sbatch
```
