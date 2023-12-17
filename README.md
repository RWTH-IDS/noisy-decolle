# Noisy Deep Continuous Local Learning (NoisyDECOLLE)

NoisyDECOLLE is a framework for analyzing the impact of various noise sources on spiking neural networks (SNNs). The supported SNNs are trained online with local learning rules following the [DECOLLE algorithm](https://www.frontiersin.org/articles/10.3389/fnins.2020.00424/full). This implementation is based on the original [DECOLLE framework](https://github.com/nmi-lab/decolle-public/tree/master/decolle).


### Create Conda Environment
The framework is tested, using python 3.9. It can be used in an conda environment.
```
conda create -n noisy-decolle python=3.9
conda activate noisy-decolle
```
Further information about installing the latest miniconda version can be found at https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html.

### Install NoisyDECOLLE
```
git clone https://github.com/RWTH-IDS/noisy-decolle.git
cd noisy-decolle
pip install -r requirements.txt
```

### Download DVS dataset
Download DvsGesture.tar.gz from https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/file/211521748942?sb=/details and place it in the noisy-decolle/ directory. Then run
```
mkdir -p data/dvsgesture
tar -xzf DvsGesture.tar.gz -C ./data/dvsgesture
mv data/dvsgesture/DvsGesture data/dvsgesture/raw
rm DvsGesture.tar.gz
```
The import of the N-MNIST dataset works on the fly.

### Run training on DVS with default parameters
```
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml
```
Due to the way the torchneuromorphic dataloader is implemented, the first training run of DVS should not be stopped before the 4th epoch. The log-file subfolder and the save directory are controlled using `--log_folder` and `--save_dir`.

### Run noisy inference on pretrained network
```
#### Spike loss
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir spikeloss_test --spike_loss 50  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless

#### Background activity
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir ba_test --ba_noise 4  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless

#### Hot pixels
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir hotpixel_test --hot_pixels 0.17  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless

#### Mismatch
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir mismatch_test --mismatch 0.2  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless

#### Quantization (AbsP)
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir quant_test --quantise_bits 5 --percentile 99  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless

#### Thermal
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --save_dir thermal_test --thermal_noise 0.005  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless
```

### Run noise-aware training
```
#### 8b Quantization of 20% of all weights (MinMax Quantization)
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --quantise_bits 8 --quant_method tensor --p_quantise 0.2

#### Thermal
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --thermal_noise 0.01
```

### Run noisy inference on networks trained with quantization aware training (QAT)
```
#### 4b Minmax puantization during training (100% of weights quantized) and testing
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --quantise_bits 4 --quant_method tensor  --no_train true --resume_from pretrained/dvsgestures_minmax_bits_4_p_1

#### Thermal noise 0.01 during training and testing
python train_lenet_decolle.py --params_file parameters/params_dvsgestures.yml --thermal_noise 0.01  --no_train true --resume_from pretrained/dvsgestures_thermal_noise_0.01
```


### Quantization options
The quantization is controlled by several parameters. All quantization methods included, use uniform quantization. The option `--quantise_bits` sets the number of bits quantized during training/testing. In the framework, we included different quantization frameworks. They can be set, using the `--quant_method` parameter, and are the following:
* `--quant_method brevitas` Absolute Maximum (AbsMax) quantization
    + Takes the maximum absolute weight value in the weight tensor of a given layer to calculate the scaling factor
    + Uses the Brevitas library for comparability with other papers
* `--quant_method float` Absolute Percentile (AbsP) quantization (https://doi.org/10.1109/ICMLA55696.2022.00243)
    + Takes the k-th percentile of absolute weights in the weight tensor of a given layer to calculate the scaling factor
    + `--percentile k` Sets the percentile (k in %)
* `--quant_method tensor` MinMax Quantization 
    + Takes the minimum and maximum weight value in the weight tensor of a given layer to calculate the scaling factor
    + Uses the Torch package's MinMaxObserver

During Quantization Aware Training, the percentage of weight values quantized can be changed using the parameter `--p_quantise p` (p ranges from 0 to 1, 1 means all weights are quantized). During testing, all weight values are quantized.

### Spike Activation Map (SAM)
Spiking Activation Maps (SAMs) are a useful tool to visualize the network's activity. Due to the high need of RAM or VRAM, the function is outsourced. It can output the spiking activity for all different input categories or just one. This is determined by the parameter `--sam label`. For single input gifs, label ranges from 1-10 for N-MNIST and 1-11 for DVS. If label is set to -1, there will be one gif for each input category. The storage location can be set using `--sam_directory`. The two prompts are added to the other input options, e.g.:
```
mkdir sam
python plot_sam.py --params_file parameters/params_dvsgestures.yml  --no_train true --resume_from pretrained/dvsgestures_trained_noiseless --sam 5 --sam_directory sam 
```
In case you run out of VRAM, the computations can be moved to the CPU by adding `--device cpu` in the command line.

### License
This project is mainly based on DECOLLE which is licensed under GPLv3 - see LICENSE.txt. Modified files have a modification notice in the header. Newly added files have a header marking them as such. 

All code within the folder "quantization" is solely based on the [Facebook AI Research Sequence-to-Sequence Toolkit](https://github.com/facebookresearch/fairseq/tree/main) which is licensed under MIT license - see quantization/LICENSE.txt. All modified code within that folder is permitted to be used under the MIT license. All other code in this repository is permitted to be used under the GPLv3 license.
