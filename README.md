# Unoffical Implementation of APOLLO

This is the unofficial implementation for the paper from my friends,  "APOLLO: SGD-like Memory, AdamW-level Performance", which is validated by them.

## Setup

```
conda env create -f environment.yml
```

Or you can setup enviroment following steps in [Q-Galore](https://github.com/VITA-Group/Q-GaLore).

## Usage

### Save optimizer memory using APOLLO optimizers

```
from galore_torch import APOLLO
# define param groups as apollo_params and non_apollo_params
param_groups = [{'params': non_apollo_params}, 
                {'params': apollo_params, 'rank': 1, 'proj': 'random', 'scale_type': 'tensor', 'scale': 128,
			  'update_proj_gap': 200, 'proj_type': 'std'}]
optimizer = APOLLO(param_groups, lr=0.01)
```

#### `rank`
- Specifies the rank of the auxiliary sub-space used for gradient scaling.
- **Default value:** 
    - `1` for APOLLO-Mini. 
    - `256` for APOLLO works well for 1B and 7B model.

#### `scale_type`
- Determines how the scaling factors are applied:
  - **`tensor`**: Applies gradient scaling at the tensor level (APOLLO-Mini).
  - **`channel`**: Applies gradient scaling at the channel level (APOLLO)

#### `scale`
- Governs the scaling factor for gradient updates. Can be tuned for better performance.
    - `128` for APOLLO-Mini by default. 
    - `1` for APOLLO by default.

### Benchmark 1: Pre-Training LLaMA on C4 dataset

We provide the command in `scripts/benchmark_c4` for pretraining LLaMA model with sizes from 60M to 7B on C4 dataset.

```
# rank = 1
# tensor: the gradient scaling factor in calculated tensor-wise (option: channel)
# projection type: random projection (option: svd)
# scale: related with rank, larger rank generally works well with smaller scale, we use 128 for rank=1

```

We also provide the pretraining scripts for LLaMA-7B with adam scripts in `scripts/benchmark_c4/llama_7b_adam.sh`.

### Benchmark 2: Pretraining LLaMA-7B model within 16GB memory

The command of training LLaMA-7B model on single GPU as provided within `scripts/single_gpu`. With 1 batch size, the following scripts can pre-train a LLaMA-7B model within 11GB memory (tested on a single A100 GPU)

## Citation

```bibtex
@misc{zhu2024apollosgdlikememoryadamwlevel,
      title={APOLLO: SGD-like Memory, AdamW-level Performance}, 
      author={Hanqing Zhu and Zhenyu Zhang and Wenyan Cong and Xi Liu and Sem Park and Vikas Chandra and Bo Long and David Z. Pan and Zhangyang Wang and Jinwon Lee},
      year={2024},
      eprint={2412.05270},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2412.05270}, 
}
```