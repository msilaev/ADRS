import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import torch.nn as nn

import argparse

from models.io import plt_pca_projections


import torch.optim as optim
from models.dataset_batch_norm import BatchData
from torch.utils.data import Dataset, DataLoader
from models.io import load_h5, upsample_wav, H5Dataset

import soundfile as sf
from torch.utils.data import ConcatDataset

from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import joblib
import librosa
import librosa.display
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import ScalarFormatter


def find_perfect_features(X_train, X_test, y_train, y_test, results_dir, model, sr, prefix):
    n_features = X_train.shape[1]
    accuracies = np.full(n_features, np.nan)
    for i in range(n_features):
        try:
            lda_i = LinearDiscriminantAnalysis()
            lda_i.fit(X_train[:, i:i+1], y_train)
            accuracies[i] = accuracy_score(y_test, lda_i.predict(X_test[:, i:i+1]))
        except Exception:
            pass

    perfect_idx = np.where(accuracies == 1.0)[0]
    print(f"Features with 100% accuracy: {len(perfect_idx)} / {n_features}")
    print(f"Indices: {perfect_idx.tolist()}")

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{prefix}_perfect_features_{model}_{sr}.txt"), 'w') as f:
        f.write(f"Features with 100% accuracy: {len(perfect_idx)} / {n_features}\n")
        f.write(f"Indices: {perfect_idx.tolist()}\n")

    colors = ['red' if a == 1.0 else 'steelblue' for a in accuracies]
    plt.figure(figsize=(12, 4))
    plt.bar(np.arange(n_features), accuracies, color=colors, width=1.0)
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1)
    plt.xlabel("Feature index", fontsize=20)
    plt.ylabel("LDA accuracy", fontsize=20)
    plt.tight_layout()
    plt.grid(axis='y')
    plt.savefig(os.path.join(results_dir, f"{prefix}_per_feature_accuracy_{model}_{sr}.pdf"))
    plt.close()


def save_classifier_artifact(logs_dir, embedding_name, model_name, sr, scaler, classifier):
    os.makedirs(logs_dir, exist_ok=True)
    artifact = {
        "model": classifier,
        "scaler": scaler,
        "label_map": {0: "enhanced", 1: "real"},
        "embedding": embedding_name,
        "source_model": model_name,
        "sr": sr,
    }
    paths = [
        os.path.join(logs_dir, f"{embedding_name}_{model_name}_classifier.joblib"),
        os.path.join(logs_dir, f"{embedding_name}_{model_name}_{sr}_classifier.joblib"),
    ]
    for path in paths:
        joblib.dump(artifact, path)
    print(f"Saved classifier artifact to {paths[0]}")


def make_parser():
  parser = argparse.ArgumentParser()
  subparsers = parser.add_subparsers(title='Commands')

  # train
  train_parser = subparsers.add_parser('train')
  train_parser.set_defaults(func=train)
  train_parser.add_argument('--model', default='gan')
  train_parser.add_argument('--train', required=True,
    help='path to h5 archive of training patches')
  train_parser.add_argument('--val', required=True,
    help='path to h5 archive of validation set patches')
  train_parser.add_argument('--batch_size', type=int, default=32,
    help='training batch size')
  train_parser.add_argument('--sr', help='high-res sampling rate',
                           type=int, default=48000)
  train_parser.add_argument('--patch_size', type=int, default=8192,
                           help='Size of patches over which the model operates')
  train_parser.add_argument('--results_dir', type=str, default="results")
  train_parser.add_argument('--logs_dir', type=str, default="logs_classifiers")
  train_parser.add_argument('--find_perfect', action='store_true',
                           help='scan each feature for 100% single-feature LDA accuracy')

  return parser

def train(args):

    if torch.cuda.is_available() :
        print("CUDA!")
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    batch_size = args.batch_size
    X_val, y_val = load_h5(args.val)

    dataset_val = BatchData(X_val, y_val, lr_mean =0, lr_std=1, hr_mean=0, hr_std=1)
    val_loader = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, drop_last=True)

    X_train, Y_train = load_h5(args.train)
    dataset_train = BatchData(X_train, Y_train, lr_mean =0, lr_std=1, hr_mean=0, hr_std=1)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=False, drop_last=True)

    batch_size = train_loader.batch_size  # Keep the batch size from DataLoader

    # Initialize StandardScaler and IncrementalPCA
    scaler = StandardScaler()
    num_splits = 11
    num_splits = 2

    transformed_data_train = []
    labels_list_train = []

    for batch in train_loader:

        features, labels = batch
        batch_size, total_length, _ = features.shape
        split_size = total_length // num_splits

        feature_split = features.reshape(batch_size*num_splits, split_size, 1)
        labels_split = labels.repeat(1,1, num_splits)
        labels_split = labels_split.reshape(batch_size*num_splits, 1,1)
        transformed_data_train.append(feature_split.view(-1, split_size))
        labels_list_train.append(labels_split.view(-1).numpy())


    transformed_data_train = np.vstack(transformed_data_train)
    labels_train = np.concatenate(labels_list_train)

    X_train, X_test, y_train, y_test = \
        train_test_split(transformed_data_train, labels_train, test_size=0.2, random_state=42)

    X_train = scaler.fit_transform(X_train)  # Fit & transform training data
    X_test = scaler.transform(X_test)  # Only transform test data

    X_val = X_val.squeeze(-1)
    y_val = y_val.squeeze(-1)


    ####################################

    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)

    y_pred = lda.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    accuracy_string = f"lda, {accuracy:.4f}\n"
    print(accuracy_string)
    os.makedirs(args.logs_dir, exist_ok=True)
    save_classifier_artifact(args.logs_dir, 'convol', args.model, args.sr, scaler, lda)

    with open(os.path.join(args.results_dir, f"conv_accuracy_lda_{args.model}_{args.sr}.txt"), 'a') as f:
        f.write(accuracy_string)

    X_test_lda = lda.transform(X_test)

    font_size_label = 30
    font_size_ticks = 22

    bins = np.linspace(-6, 6, 30)
    plt.figure(figsize=(8, 5))
    plt.hist(X_test_lda[y_test == 0], bins=bins, alpha=0.6, label="fake", color="blue")
    plt.hist(X_test_lda[y_test == 1], bins=bins, alpha=0.6, label="real", color="red")
    plt.xlabel("LDA Projection", fontsize=font_size_label)
    plt.ylabel("Number", fontsize=font_size_label)

    ax = plt.gca()
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax.yaxis.offsetText.set_fontsize(font_size_label)

    plt.xticks(2*np.arange(-3, 4), fontsize=font_size_label)
    plt.yticks(200 * np.arange(0, 6), fontsize=font_size_label)
    plt.yticks(fontsize=font_size_label)
    plt.legend(fontsize=font_size_label)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{args.results_dir}/conv_hist_lda_{args.model}_{args.sr}.pdf")
    plt.show()

        # LDA coefficient plot: which embedding dimensions are most discriminative
    coef = lda.coef_[0]
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(coef)), coef)
    plt.xlabel("Embedding dimension", fontsize=font_size_label)
    plt.ylabel("LDA weight", fontsize=font_size_label)
    plt.xticks(fontsize=font_size_ticks)
    plt.yticks(fontsize=font_size_ticks)
    plt.tight_layout()
    plt.grid()
    plt.savefig(f"{args.results_dir}/conv_lda_coef_{args.model}_{args.sr}.pdf")
    plt.show()

    if args.find_perfect:
        find_perfect_features(X_train, X_test, y_train, y_test,
                              args.results_dir, args.model, args.sr, "conv")

    # -------------------------------------------------------------------------
    # PCA 2D visualization
    # -------------------------------------------------------------------------
    print("Running PCA 2D ...")
    X_all = np.vstack([X_train, X_test])
    Y_all = np.concatenate([y_train, y_test]).astype(int)
    n_real = (Y_all == 1).sum()
    n_fake = (Y_all == 0).sum()
    pca2 = PCA(n_components=2, random_state=42)
    X_pca = pca2.fit_transform(X_all)
    var = pca2.explained_variance_ratio_
    font = 13
    fig, axes = plt.subplots(1, 2, figsize=(9, 12))
    ax = axes[0]
    ax.scatter(X_pca[Y_all == 1, 0], X_pca[Y_all == 1, 1],
               alpha=0.3, s=8, color='red', label=f'real (n={n_real})', rasterized=True)
    ax.scatter(X_pca[Y_all == 0, 0], X_pca[Y_all == 0, 1],
               alpha=0.3, s=8, color='blue', label=f'enhanced (n={n_fake})', rasterized=True)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=font)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)", fontsize=font)
    ax.set_title(f"Convol PCA — {args.model.upper()} {args.sr//1000} kHz", fontsize=font+1, fontweight='bold')
    ax.legend(fontsize=font-2, markerscale=3)
    ax.tick_params(labelsize=font-2)
    ax.grid(alpha=0.3)
    ax = axes[1]
    bins = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 60)
    ax.hist(X_pca[Y_all == 0, 0], bins=bins, alpha=0.6, color='blue', label='enhanced')
    ax.hist(X_pca[Y_all == 1, 0], bins=bins, alpha=0.6, color='red', label='real')
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)", fontsize=font)
    ax.set_ylabel("Count", fontsize=font)
    ax.set_title("PC1 distribution", fontsize=font+1, fontweight='bold')
    ax.legend(fontsize=font-2)
    ax.tick_params(labelsize=font-2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"conv_pca2d_{args.model}_{args.sr}.pdf")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved → {out}")

def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()
