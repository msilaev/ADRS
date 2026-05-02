Audio Super Resolution Using Neural Networks
============================================

This repository contains PyTorch implementation of the supervised learning and Generative Adversarial Network (GAN) approach to the audio upsampling.
Two upsampling tasks are considred:  4 to 16 kHz and 16 to 48 kHz. The supervised model (AudioUnet)architecture is taken from paper

```
S. Birnbaum, V. Kuleshov, Z. Enam, P. W.. Koh, and S. Ermon, Temporal FiLM: Capturing Long-Range Sequence Dependencies with Feature-Wise Modulations. NeurIPS 2019
V. Kuleshov, Z. Enam, and S. Ermon. Audio Super Resolution Using Neural Networks. ICLR 2017 (Workshop track)
```
The starting point was the realization of supervised learning upsampling using TensorFlow 
```
https://github.com/kuleshov/audio-super-res
```


The generative model MU-GAN general idea is taken from paper 

```
S. Kim and V. Sathe, Bandwidth extension on raw audio via generative adversarial networks, arXiv preprint arXiv:1903.09027
```
Model architecture is completed by the particular values of weights in convolutional layers. Also, the training protocol is 
suggested to stabilize adversarial training. No autoencoder fearture loss function suggested in the original paper is used. 




## Setup

### Requirements

This repository was tested on Ubuntu 20.04

- python=3.11.0
- cudatoolkit
- cudnn
- torch
- torchaudio
- numpy
- scipy
- librosa
- h5py
- matplotlib

### Project structure

The following diagram illustrates the project structure
```
ADSRpaper/
├── data/
│   ├── vctk/
│   │   ├── prep_vctk_multispeaker.py
│   │   ├── prep_vctk_singlespeaker.py
│   │   ├── multispeaker/
│   │   │   └── generate_upsampling_dataset.sh
│   │   └── speaker1/
│   │       └── generate_upsampling_dataset.sh
├── models/
│   ├── gan.py
│   ├── audiounet.py
│   └── audiotfil.py
├── src_train/
│   ├── run_training_audiounet_16_r_4_multispeaker.sh
│   ├── run_training_audiounet_16_r_4_singlespeaker.sh
│   ├── run_training_audiounet_48_r_3_singlespeaker.sh
│   ├── run_training_gan_16_r_4_multispeaker.sh
│   ├── run_training_gan_16_r_4_singlespeaker.sh
│   ├── run_training_gan_48_r_3_multispeaker.sh
│   ├── run_training_gan_48_r_3_singlespeaker.sh
│   ├── run_training_ganAlt3_16_r_4_multispeaker.sh
│   ├── run_training_ganAlt5_16_r_4_multispeaker.sh
│   ├── run_training_ganAlt3_48_r_3_multispeaker.sh
│   ├── run_training_ganAlt5_48_r_3_multispeaker.sh
│   ├── run_training_genDecoupled_16_r_4_multispeaker.sh
│   ├── run_training_genDecoupled_16_r_4_singlespeaker.sh
│   ├── run_training_genDecoupled_48_r_3_multispeaker.sh
│   └── run_training_genDecoupled_48_r_3_singlespeaker.sh
├── src_test/
│   ├── run_generate_examples.sh
│   ├── run_eval_scores.sh
│   └── run_plt_hist.sh
├── src_inference/
│   └── inference48.sh
├── src_plt/
│   └── run_plt_learning_curves.sh
├── logs/
│   ├── multispeaker/
│   ├── singlespeaker/
│   ├── test-other/
│   └── metrics_summary.txt
├── results/
│   ├── learning_curves/
│   ├── test-other/
└── environment.yaml 
```

### Installation

1. Either clone the repository 
   ```bash
   git clone ...
   cd adversarial-bandwidth-expansion
   conda env create -f environment.yaml 
   conda activate audio-enh-supervise
   ```
   or alternatively unpack the zip achive and  
   ```bash   
   cd adversarial-bandwidth-expansion
   ```    

2. Create virtual environment
   ```bash   
   conda env create -f environment.yaml
   conda activate audio-enh-supervise
   ```

### Retrieving audio data 

The `./data` subfolder contains make file for downloading and unpacking the VCTK speech dataset
   ```bash   
   cd data/vctk
   make
   ```

### Preparing dataset for training upsampling models  

Datasets here consists of pairs of high and low resolution sound patches of the fixed length.
This is handled by `./data/vctk/prep_vctk_multispeaker.py` and `./data/vctk/prep_vctk_singlespeaker.py` modules. 
The output of the data preparation step are two `.h5` archives containing, respectively, the training and 
validation pairs of high/low resolution sound patches.

They can be run using following scripts.

To prepare dataset for Speaker-1 task (training and testing using single speaker)
```
cd data/vctk/speaker1
./generate_upsampling_dataset.sh
```

To prepare dataset for Speaker-1 task (training and testing using single speaker)
```
cd data/vctk/multispeaker
./generate_upsampling_dataset.sh
```

## Training models
Models are defined in `./models/gan.py`, `./models/audiounet.py`, `./models/audiotfil.py`. The latter one is not in use.
Training scripts are in `./src_train` folder. 
Model checkpoints and loss are saved in corresponding subfolders `./logs/multispeaker/` and `./logs/singlespeaker/`.
Starting training of different models is done as follows
```
cd src_train
chmod +x run_training_audiounet_16_r_4_multispeaker.sh
```
and similarly with other models.

Models with supervised learning Audiounet
```
./run_training_audiounet_16_r_4_multispeaker.sh
```
```
./run_training_audiounet_16_r_4_singlespeaker.sh
```
```
./run_training_audiounet_48_r_3_singlespeaker.sh
```
Models with usual GAN
```
./run_training_gan_16_r_4_multispeaker.sh
```
```
./run_training_gan_16_r_4_singlespeaker.sh
```
```
./run_training_gan_48_r_3_multispeaker.sh
```

```
./run_training_gan_48_r_3_singlespeaker.sh
```
Models with usual GAN and scheduled Discriminator updates
```
./run_training_ganAlt3_16_r_4_multispeaker.sh
```
```
./run_training_ganAlt5_16_r_4_multispeaker.sh
```
```
./run_training_ganAlt3_48_r_3_multispeaker.sh
```
```
./run_training_ganAlt5_48_r_3_multispeaker.sh
```

Models with decoupled Generators of GAN
```
./run_training_genDecoupled_16_r_4_multispeaker.sh
```
```
./run_training_genDecoupled_16_r_4_singlespeaker.sh
```
```
./run_training_genDecoupled_48_r_3_multispeaker.sh
```
```
run_training_genDecoupled_48_r_3_singlespeaker.sh
```

## Model Evaluation
1. Models can be evaluated by plotting example spectrograms and audio files with origonal and donwsampled-restored samples. 
This is done for VCTK test set samples specified in the file `data/vctk/multispeaker/val-files-short-short.txt` using the command 
```
cd src_test
./run_generate_examples.sh
```

2. To calculate the overall SNR and LSD metrics for the whole VCTK test set for each model use 
```
cd src_test
./run_get_metrics.sh
```
This will generate table in text file `logs/metrics_summary.txt`

## Upsampling of arbitrary audio sets

To upsample arbitrary set of audio files 

1. Put audio files in folder `./data/test-other`

2. Upsampling using GAN, Audiounet, or both in one sitting 

```
cd src_inference
./inference48.sh gan
./inference48.sh audiounet
./inference48.sh both
```

These commands will read 16 kHz audio files in the `./data/test-other` and write the 
processed files with same names into folders `./results/test-other` and `./results/test-other` 

## Plotting learning curves

```
cd src_plt
./run_plt_learning_curves.sh
```

This will read loss text files from corresponding subfolders in `./logs/test-other` plot learning curves and save pictures
to the folder `./result/learning_curves`

## Classifier-based Quality Assessment

The `src_classify/` folder contains scripts for evaluating how distinguishable
generated audio is from real audio using LDA-based classifiers on two feature spaces:
convolutional discriminator embeddings and mel spectrograms.

### Pipeline

**Step 1 — generate raw patch H5 archive**

For each audio file: downsample to low-res, upsample with the generator, write HR patch
(label=1) and generated patch (label=0) to an HDF5 archive.

```bash
cd src_classify
python prep_vctk_patches_16.py \
    --file-list ../data/vctk/multispeaker/val-files.txt \
    --in-dir /path/to/vctk \
    --out patches_16.h5 \
    --scale 4 --dimension 48000 --stride 48000 \
    --sr 16000 --patch 48000 \
    --model gan_melLoss \
    --checkpoint_path ../logs/multispeaker/sr16000/.../generator.pth
```

Use `prep_vctk_patches_48.py` for the 16→48 kHz task (`--scale 3 --sr 48000`).

**Step 2a — extract convolutional embeddings** (requires discriminator checkpoint)

```bash
python prep_vctk_convol_embed_16.py \
    --input patches_16.h5 \
    --out embed_convol_16.h5 \
    --len <number_of_samples_in_patches_16.h5> \
    --checkpoint_path ../logs/multispeaker/sr16000/.../discriminator.pth
```

**Step 2b — extract mel spectrogram embeddings** (no checkpoint needed)

```bash
python prep_vctk_stft_embed_16.py \
    --input patches_16.h5 \
    --out embed_mel_16.h5 \
    --len <number_of_samples> \
    --sr 16000
```

**Step 3 — train LDA classifier on convolutional embeddings**

```bash
# 16 kHz
python convol_PCA_accuracy_16.py train \
    --train embed_convol_16_train.h5 \
    --val   embed_convol_16_val.h5 \
    --model gan_melLoss --sr 16000 \
    --results_dir results --logs_dir logs_classifiers

# 48 kHz (single H5, no separate val)
python convol_PCA_accuracy_48.py train \
    --train embed_convol_48.h5 \
    --model gan_melLoss --sr 48000 \
    --results_dir results --logs_dir logs_classifiers
```

**Step 3 — analyze mel spectrogram differences**

```bash
python analyze_mel_differences.py \
    --input embed_mel_16.h5 \
    --model gan_melLoss --sr 16000 \
    --results_dir results

python partial_band_accuracy.py \
    --input embed_mel_16.h5 \
    --model gan_melLoss --sr 16000 \
    --results_dir results
```

### Outputs

| Script | Outputs |
|---|---|
| `convol_PCA_accuracy_*.py` | LDA histogram PDF, LDA weight-vector PDF, 2D PCA scatter PDF, `.joblib` classifier artifact |
| `analyze_mel_differences.py` | Mean spectra, real−fake difference, per-bin LDA accuracy, LDA weight vector PDFs + PNGs, text summary |
| `partial_band_accuracy.py` | Accuracy-vs-cutoff-frequency curve PDF + PNG, text table |

### Additional dependencies

```
scikit-learn
joblib
```

