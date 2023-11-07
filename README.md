# Noisy Deep Continuous Local Learning (NoisyDECOLLE)

NoisyDECOLLE is a framework for analyzing the impact of various noise sources on spiking neural networks (SNNs). The supported SNNs are trained online with local learning rules following the [DECOLLE algorithm](https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full). This implementation is based on the original [DECOLLE framework](https://github.com/nmi-lab/decolle-public/tree/master/decolle).

### Install NoisyDECOLLE
```
git clone https://github.com/RWTH-IDS/noisy-decolle.git
cd noisy-decolle
pip install -r requirements.txt
```

### Run training on DVS with default parameters
```
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml
```

### Run noisy inference on pretrained network
```
#### Spike loss
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir spikeloss_test --spike_loss $noise  --no_train true --resume_from results/pretrained

#### Background activity
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir ba_test --ba_noise $noise  --no_train true --resume_from results/pretrained

#### Hot pixels
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir hotpixel_test --hot_pixels $noise  --no_train true --resume_from results/pretrained

#### Mismatch
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir mismatch_test --mismatch $noise  --no_train true --resume_from results/pretrained

#### Quantization
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir quant_test --quantise_test $noise --percentile 99  --no_train true --resume_from results/pretrained

#### Thermal
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir thermal_test --thermal_noise $noise  --no_train true --resume_from results/pretrained
```

### Run noise-aware training
```
#### 8b Quantization of 20% of all weights
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --quantise_training 8 --p_quantise 0.2

#### Thermal
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --thermal_noise $noise
```

### License
This project is mainly based on DECOLLE which is licensed under GPLv3 - see LICENSE.txt. Modified files have a modification notice in the header. Newly added files have a header marking them as such. 

All code within the folder "quantization" is solely based on the [Facebook AI Research Sequence-to-Sequence Toolkit](https://github.com/facebookresearch/fairseq/tree/main) which is licensed under MIT license - see quantization/LICENSE.txt. All modified code within that folder is permitted to be used under the MIT license. All other code in this repository is permitted to be used under the GPLv3 license.