import gc
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import argparse

from models.dataset_batch_norm import BatchData
from torch.utils.data import Dataset, DataLoader
from models.io import load_h5

from sklearn.decomposition import IncrementalPCA
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from matplotlib.ticker import ScalarFormatter

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

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
  train_parser.add_argument('--batch_size', type=int, default=32,
    help='training batch size')
  train_parser.add_argument('--sr', help='high-res sampling rate',
                           type=int, default=48000)
  train_parser.add_argument('--patch_size', type=int, default=8192,
                           help='Size of patches over which the model operates')
  train_parser.add_argument('--logs_dir', type=str, default="logs")
  train_parser.add_argument('--results_dir', type=str, default="results")
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
    X_train, Y_train = load_h5(args.train)
    dataset_train = BatchData(X_train, Y_train, lr_mean =0, lr_std=1, hr_mean=0, hr_std=1)
    train_loader = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, drop_last=True)
    batch_size = train_loader.batch_size  # Keep the batch size from DataLoader

    # Initialize StandardScaler and IncrementalPCA
    scaler = StandardScaler()
    num_splits = 5

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

    # Split into train/test sets
    X_train, X_test, y_train, y_test = \
        train_test_split(transformed_data_train, labels_train, test_size=0.2)

    X_train = scaler.fit_transform(X_train)  # Fit & transform training data
    X_test = scaler.transform(X_test)  # Only transform test data
    ##############################################

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
    plt.yticks(500 * np.arange(0, 7), fontsize=font_size_label)
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

def main():

  torch.cuda.empty_cache()
  gc.collect()

  parser = make_parser()
  args = parser.parse_args()
  args.func( args)

if __name__ == '__main__':
  main()
