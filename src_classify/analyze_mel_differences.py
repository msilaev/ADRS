"""
Analyze which mel spectrogram frequency bands differ between real and enhanced audio,
as encoded in the time-averaged mel embedding H5 files produced by src_embeddings.

Each feature dimension i in the embedding corresponds to mel frequency bin i
(256 bins, time-averaged with AdaptiveAvgPool2d).

Outputs per model/SR:
  mel_mean_spectra_{model}_{sr}.pdf      - mean mel spectrum real vs fake
  mel_difference_{model}_{sr}.pdf        - real minus fake difference in dB
  mel_per_bin_accuracy_{model}_{sr}.pdf  - per-bin single-feature LDA accuracy vs Hz
  mel_analysis_{model}_{sr}.txt          - summary statistics
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import librosa
from scipy.ndimage import gaussian_filter1d
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.io import load_h5


def make_parser():
    parser = argparse.ArgumentParser(
        description='Analyze mel spectrogram differences learned by the classifier')
    parser.add_argument('--input', required=True,
                        help='Path to H5 mel embedding file')
    parser.add_argument('--model', default='gan',
                        help='Model name (for output filenames)')
    parser.add_argument('--sr', type=int, default=48000,
                        help='Sample rate used when computing the mel spectrogram')
    parser.add_argument('--n_mels', type=int, default=256,
                        help='Number of mel bins in the embedding')
    parser.add_argument('--n_fft', type=int, default=4096,
                        help='FFT size used when computing the mel spectrogram')
    parser.add_argument('--results_dir', type=str, default='results',
                        help='Directory to save output figures and text')
    return parser


def mel_bin_frequencies(n_mels, sr, n_fft):
    """Return center frequency in Hz for each mel bin."""
    return librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2, htk=False)


def per_bin_lda_accuracy(X_train, X_test, y_train, y_test):
    """Run a single-feature LDA for each mel bin and return accuracy array."""
    n_bins = X_train.shape[1]
    accuracies = np.full(n_bins, np.nan)
    for i in range(n_bins):
        try:
            lda = LinearDiscriminantAnalysis()
            lda.fit(X_train[:, i:i+1], y_train)
            accuracies[i] = accuracy_score(y_test, lda.predict(X_test[:, i:i+1]))
        except Exception:
            pass
    return accuracies


def set_linear_xaxis(ax, sr):
    """Linear x-axis from 0 to sr/2 with clean kHz tick marks."""
    nyq = sr // 2
    step = max(1000, round(nyq / 6 / 1000) * 1000)
    ticks = list(range(0, nyq + 1, step))
    if ticks[-1] != nyq:
        ticks.append(nyq)
    ax.set_xlim(0, nyq)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, _: f"{int(x)}" if x < 1000 else f"{int(x)//1000}k"))


def analyze(args):
    os.makedirs(args.results_dir, exist_ok=True)
    tag = f"{args.model}_{args.sr}"

    # -----------------------------------------------------------------
    # Load embeddings: X shape (N, n_mels, 1), Y shape (N, 1, 1)
    # -----------------------------------------------------------------
    X, Y = load_h5(args.input)
    X = X.squeeze(-1)   # (N, n_mels)
    Y = Y.squeeze()     # (N,)  — 0=fake/enhanced, 1=real

    # Drop the last mel bin: its right edge lands exactly at Nyquist,
    # capturing model aliasing artifacts rather than speech content.
    X = X[:, :-1]

    freqs = mel_bin_frequencies(args.n_mels - 1, args.sr, args.n_fft)

    n_real = int((Y == 1).sum())
    n_fake = int((Y == 0).sum())
    print(f"Samples — real: {n_real}, fake: {n_fake}")

    # -----------------------------------------------------------------
    # Per-class mean mel spectrum
    # -----------------------------------------------------------------
    mean_real = X[Y == 1].mean(axis=0)
    mean_fake = X[Y == 0].mean(axis=0)
    std_real  = X[Y == 1].std(axis=0)
    std_fake  = X[Y == 0].std(axis=0)
    diff      = mean_real - mean_fake  # positive = real has more energy

    # -----------------------------------------------------------------
    # Per-bin LDA accuracy (train/test split for honest evaluation)
    # -----------------------------------------------------------------
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    accuracies = per_bin_lda_accuracy(X_train, X_test, y_train, y_test)
    perfect_idx  = np.where(accuracies == 1.0)[0]
    perfect_freqs = freqs[perfect_idx]

    # Full LDA on all mel bins — weight vector reveals which bins drive the decision
    lda_full = LinearDiscriminantAnalysis()
    lda_full.fit(X_train, y_train)
    full_acc = accuracy_score(y_test, lda_full.predict(X_test))
    lda_weights = lda_full.coef_[0]   # shape (n_mels,); positive → pushes toward real

    # -----------------------------------------------------------------
    # Save text summary
    # -----------------------------------------------------------------
    top5_diff_idx  = np.argsort(np.abs(diff))[::-1][:5]
    top5_acc_idx   = np.argsort(accuracies)[::-1][:5]

    summary_path = os.path.join(args.results_dir, f"mel_analysis_{tag}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Model: {args.model}   SR: {args.sr} Hz   n_mels: {args.n_mels}\n")
        f.write(f"Samples — real: {n_real}, fake: {n_fake}\n\n")

        f.write("Top 5 bins by |real - fake| difference (dB):\n")
        for idx in top5_diff_idx:
            f.write(f"  bin {idx:3d}  {freqs[idx]:7.1f} Hz  diff={diff[idx]:+.3f} dB\n")

        f.write("\nTop 5 bins by single-feature LDA accuracy:\n")
        for idx in top5_acc_idx:
            f.write(f"  bin {idx:3d}  {freqs[idx]:7.1f} Hz  acc={accuracies[idx]:.4f}\n")

        f.write(f"\nBins with 100% accuracy: {len(perfect_idx)} / {args.n_mels}\n")
        for idx, hz in zip(perfect_idx, perfect_freqs):
            f.write(f"  bin {idx:3d}  {hz:7.1f} Hz\n")

        f.write(f"\nFull LDA accuracy (all {args.n_mels} bins): {full_acc:.4f}\n")
        top5_w = np.argsort(np.abs(lda_weights))[::-1][:5]
        f.write("Top 5 bins by |LDA weight|:\n")
        for idx in top5_w:
            f.write(f"  bin {idx:3d}  {freqs[idx]:7.1f} Hz  weight={lda_weights[idx]:+.4f}\n")

    print(f"Saved summary → {summary_path}")

    # -----------------------------------------------------------------
    # Plot 1: mean mel spectrum real vs fake
    # -----------------------------------------------------------------
    font = 18
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(freqs, mean_real, color='red',  label='real',     linewidth=1.5)
    ax.plot(freqs, mean_fake, color='blue', label='enhanced', linewidth=1.5)
    ax.fill_between(freqs, mean_real - std_real, mean_real + std_real,
                    alpha=0.15, color='red')
    ax.fill_between(freqs, mean_fake - std_fake, mean_fake + std_fake,
                    alpha=0.15, color='blue')
    set_linear_xaxis(ax, args.sr)
    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel("Mel energy (dB)", fontsize=font)
    ax.tick_params(labelsize=font - 4)
    ax.legend(fontsize=font)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"mel_mean_spectra_{tag}.pdf")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")

    # -----------------------------------------------------------------
    # Plot 2: real − fake difference
    # -----------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.fill_between(freqs, diff, 0,
                    where=(diff > 0), alpha=0.6, color='red',  label='real > enhanced')
    ax.fill_between(freqs, diff, 0,
                    where=(diff < 0), alpha=0.6, color='blue', label='enhanced > real')
    set_linear_xaxis(ax, args.sr)
    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel("Mean difference (dB)", fontsize=font)
    ax.tick_params(labelsize=font - 4)
    ax.legend(fontsize=font)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"mel_difference_{tag}.pdf")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")

    # -----------------------------------------------------------------
    # Plot 3: per-bin LDA accuracy vs frequency
    # -----------------------------------------------------------------
    colors = ['red' if a == 1.0 else 'steelblue' for a in accuracies]
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(freqs, accuracies, color=colors, width=np.diff(freqs, append=freqs[-1]))
    ax.axhline(1.0, color='red', linestyle='--', linewidth=1)
    set_linear_xaxis(ax, args.sr)
    ax.set_xlabel("Frequency (Hz)", fontsize=font)
    ax.set_ylabel("LDA accuracy", fontsize=font)
    ax.tick_params(labelsize=font - 4)
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"mel_per_bin_accuracy_{tag}.pdf")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved → {out}")


    # -----------------------------------------------------------------
    # Plot 4: full LDA weight vector vs frequency (Gaussian-smoothed only)
    # -----------------------------------------------------------------
    sigma = 3
    weights_smooth = gaussian_filter1d(lda_weights, sigma=sigma)

    bar_widths    = np.diff(freqs, append=freqs[-1])
    smooth_colors = ['#E94F37' if w > 0 else '#3A9BDC' for w in weights_smooth]

    _f = 16  # match font size of partial_band_accuracy.py (slide 12 companion plot)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(freqs, weights_smooth, width=bar_widths, color=smooth_colors, align='edge')
    ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax.set_ylabel(f"LDA weight (Gaussian σ={sigma})", fontsize=_f)
    ax.set_xlabel("Frequency (Hz)", fontsize=_f)
    ax.tick_params(labelsize=_f - 2)
    ax.grid(True, axis='y', alpha=0.3)
    set_linear_xaxis(ax, args.sr)
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"mel_lda_weights_{tag}.pdf")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out}")

    # -----------------------------------------------------------------
    # Plot 5: combined mel-difference + LDA weight (shared x-axis)
    # -----------------------------------------------------------------
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                                          gridspec_kw={'hspace': 0.08})

    # top: mel difference
    ax_top.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_top.fill_between(freqs, diff, 0,
                        where=(diff > 0), alpha=0.6, color='red',  label='real > enhanced')
    ax_top.fill_between(freqs, diff, 0,
                        where=(diff < 0), alpha=0.6, color='blue', label='enhanced > real')
    ax_top.set_ylabel("Mel diff (dB)", fontsize=font - 2)
    ax_top.tick_params(labelsize=font - 4)
    ax_top.legend(fontsize=font - 4, loc='upper left')
    ax_top.grid(True, alpha=0.3)

    # bottom: Gaussian-smoothed LDA weights
    ax_bot.bar(freqs, weights_smooth, width=bar_widths, color=smooth_colors, align='edge')
    ax_bot.axhline(0, color='black', linewidth=0.8, linestyle='--')
    ax_bot.set_ylabel(f"LDA weight (σ={sigma})", fontsize=font - 2)
    ax_bot.set_xlabel("Frequency (Hz)", fontsize=font)
    ax_bot.tick_params(labelsize=font - 4)
    ax_bot.grid(True, axis='y', alpha=0.3)

    for ax in (ax_top, ax_bot):
        set_linear_xaxis(ax, args.sr)

    fig.suptitle(
        f"{args.model}  {args.sr} Hz   LDA acc={full_acc:.1%}   "
        f"Red: toward real   Blue: toward enhanced",
        fontsize=font - 2)
    plt.tight_layout()
    out = os.path.join(args.results_dir, f"mel_combined_{tag}.pdf")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.savefig(out.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved -> {out}")


if __name__ == '__main__':
    args = make_parser().parse_args()
    analyze(args)
