<h1 align="center"> Patch-I<sup>2</sup>SB: Patch-based Image-to-Image Schrödinger Bridge </h1>

PyTorch implementation of **Patch-I<sup>2</sup>SB**. Patch-I<sup>2</sup>SB is a combination of [I2SB](https://github.com/NVlabs/I2SB) and [Patch Diffusion](https://github.com/Zhendong-Wang/Patch-Diffusion) for fast and data-efficient image-to-image translation.

## What has changed from the original I2SB?

* Data corruption process code for restoration tasks is removed. This code is only for general image-to-image translation tasks.
* Dataloader is changed from lmbd to customized dataloader. Use `dataset/dataloader.py` brought from [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)'s dataloader. It also supports the CT dicom file extension, and other medical imaging modalities will be supported later.
* Image512Net class is added in `network.py` for 512 x 512 resolution image training.
* Evaluation metric for validation is changed from accuracy to (RMSE, PSNR, and SSIM).
* Patch diffusion technique is applied. You can use it by adding the flag `--run-patch` to the [training options](https://github.com/cychoi97/Patch-I2SB#training).

### Caution

For general tasks, I<sup>2</sup>SB authors recommanded adding the flag `--cond-x1` to the [training options](https://github.com/NVlabs/I2SB#training) to overcome the large information loss in the new priors.


## Installation

This code is developed with Python3, and we recommend PyTorch >=1.11.
Install the other packages in `requirements.txt` following:
```bash
pip install -r requirements.txt
```


## Data and results

Use the flag `--dataset-dir $DATA_DIR` to specify the dataset directory. Also, use the flags `--src $SRC` and `--trg $TRG` to specify the corrupt and clean data folder name. **Images should be normalized to [-1,1].** All training and sampling results will be stored in `results`. The overall file structures are:
```text
$DATA_DIR/                           # dataset directory
    ├── train/                       # train folder
    │     ├── $SRC/                  # corrupt data folder name --src $SRC
    │     │     ├── ...              # sub folder
    │     │     │    ├── 0001.png    # image file
    │     │     │    ├── 0002.png
    │     │     │    └── 0003.png
    │     │     ├── ...
    │     │     └── ...
    │     └── $TRG/                  # clean data folder name --trg $TRG
    │           ├── ...
    │           ├── ...
    │           └── ...
    ├── valid/                       # valid folder
    └── test/                        # test folder

results/
├── $NAME/                               # experiment ID set in train.py --name $NAME
│   ├── $NUM_ITR.pt                      # latest checkpoint: network, ema, optimizer
│   ├── options.pkl                      # full training options
│   └── samples_nfe$NFE_iter$NUM_ITR/    # images reconstructed from sample.py --nfe $NFE --num-itr $NUM_ITR
│       └── recon.pt
├── ...
```


## Training

To train an **Patch-I<sup>2</sup>SB** on a single node, run
```bash
python train.py --name $NAME --n-gpu-per-node $N_GPU \
    --src $SRC --trg $TRG --dataset-dir $DATA_DIR \
    --batch-size $BATCH --microbatch $MICRO_BATCH [--ot-ode] \
    --beta-max $BMAX --log-dir $LOG_DIR [--log-writer $LOGGER] [--run-patch]
```
where `NAME` is the experiment ID, `N_GPU` is the number of GPUs on each node, `DATA_DIR` is the path to the aligned dataset, `BMAX` determines the noise scheduling. The default training on 32GB V100 GPU uses `BATCH=256` and `MICRO_BATCH=2`. If your GPUs have less than 32GB, consider lowering `MICRO_BATCH` or using smaller network.

Add `--ot-ode` for optionally training an OT-ODE model, _i.e.,_ the limit when the diffusion vanishes. By defualt, the model is discretized into 1000 steps; you can change it by adding `--interval $INTERVAL`.
Note that we initialize the network with [ADM](https://github.com/openai/guided-diffusion) ([256x256_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt) and [512x512_diffusion_uncond.pt](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/512x512_diffusion.pt)), which will be automatically downloaded to `data/` at first call.

Images and losses can be logged with either tensorboard (`LOGGER="tensorboard"`) or W&B (`LOGGER="wandb"`) in the directory `LOG_DIR`. To [autonamtically login W&B](https://docs.wandb.ai/quickstart#set-up-wb), specify additionally the flags `--wandb-api-key $WANDB_API_KEY --wandb-user $WANDB_USER` where `WANDB_API_KEY` is the unique API key (about 40 characters) of your account and `WANDB_USER` is your user name.

To resume previous training from the checkpoint, add the flag `--ckpt $CKPT`.

To run patch-based training, add the flag `--run-patch`.


## Citation

```
@article{liu2023i2sb,
  title={I{$^2$}SB: Image-to-Image Schr{\"o}dinger Bridge},
  author={Liu, Guan-Horng and Vahdat, Arash and Huang, De-An and Theodorou, Evangelos A and Nie, Weili and Anandkumar, Anima},
  journal={arXiv preprint arXiv:2302.05872},
  year={2023},
}

@article{wang2023patch,
  title={Patch Diffusion: Faster and More Data-Efficient Training of Diffusion Models},
  author={Wang, Zhendong and Jiang, Yifan and Zheng, Huangjie and Wang, Peihao and He, Pengcheng and Wang, Zhangyang and Chen, Weizhu and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2304.12526},
  year={2023}
}
```

## Acknowledgement

This code is heavily brought from [I<sup>2</sup>SB](https://github.com/NVlabs/I2SB) and [Patch Diffusion](https://github.com/Zhendong-Wang/Patch-Diffusion).

`dataloader.py` is inspired by [stylegan2-ada-pytorch](https://github.com/NVlabs/stylegan2-ada-pytorch)'s `dataset.py`.


## License
Copyright © 2023, NVIDIA Corporation. All rights reserved.

This work is made available under the Nvidia Source Code License-NC.

The model checkpoints are shared under [CC-BY-NC-SA-4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). If you remix, transform, or build upon the material, you must distribute your contributions under the same license as the original.

For business inquiries, please visit our website and submit the form: [NVIDIA Research Licensing](https://www.nvidia.com/en-us/research/inquiries/).
