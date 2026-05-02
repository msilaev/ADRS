"""
Partial-band LDA accuracy experiment.

Simulates what classification accuracy would be if the SR model only
generated up to a given output frequency (cutoff_hz), by restricting
the mel embedding to bins <= cutoff_hz before fitting LDA.

For a 16kHz model (SCA=4):
  LR Nyquist  = 2 kHz  (input band limit)
  HR Nyquist  = 8 kHz  (full output band)

For a 48kHz model (SCA=3):
  LR Nyquist  ≈ 8 kHz
  HR Nyquist  = 24 kHz

Outputs (saved to results_dir):
  partial_band_accuracy_{model}_{sr}.txt   — accuracy at each cutoff
  partial_band_accuracy_{model}_{sr}.pdf   — accuracy vs cutoff frequency curve
"""

import os
import sys
import argparse

import numpy as np
import matplotlib.pyplot as plt
import librosa
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.io import load_h5


def make_parser():
    parser = argparse.ArgumentParser(
        description='LDA accuracy vs frequency band cutoff')
    parser.add_argument('--input',       required=True,
                        help='Path to H5 mel embedding file')
    parser.add_argument('--model',       default='gan',
                        help='Model name (for output filenames)')
    parser.add_argument('--sr',          type=int, default=16000,
                        help='Sample rate used when computing the mel spectrogram')
    parser.add_argument('--n_mels',      type=int, default=256)
    parser.add_argument('--n_fft',       type=int, default=4096)
    parser.add_argument('--results_dir', type=str, default='results')
    return parser


def mel_bin_frequencies(n_mels, sr, n_fft):
    return librosa.mel_frequencies(n_mels=n_mels, fmin=0.0, fmax=sr / 2, htk=False)


def lda_accuracy(X, Y, cutoff_bin):
    """Run LDA using only mel bins 0..cutoff_bin (inclusive)."""
    X_cut = X[:, :cutoff_bin + 1]
    scaler = StandardScaler()
    X_train, X_test, y_train, y_test = train_test_split(
        X_cut, Y, test_size=0.2, random_state=42)
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train, y_train)
    return accuracy_score(y_test, lda.predict(X_test))


def main():
    args = make_parser().parse_args()
    os.makedirs(args.results_dir, exist_ok=True)

    X, Y = load_h5(args.input)
    X = X.squeeze(-1)   # (N, n_mels)
    Y = Y.squeeze()

    freqs = mel_bin_frequencies(args.n_mels, args.sr, args.n_fft)
    nyquist = args.sr / 2
    lr_nyquist = nyquist / (4 if args.sr == 16000 else 3)

    n_real = int((Y == 1).sum())
    n_fake = int((Y == 0).sum())
    print(f"Samples — real: {n_real}, fake: {n_fake}")
    print(f"SR: {args.sr} Hz  |  LR Nyquist: {lr_nyquist:.0f} Hz  |  HR Nyquist: {nyquist:.0f} Hz")

    # Evaluate at every mel bin cutoff (i.e., include bins 0..k)
    cutoff_bins = np.arange(0, args.n_mels)
    cutoff_freqs = freqs[cutoff_bins]
    accuracies = []

    for k in cutoff_bins:
        acc = lda_accuracy(X, Y, k)
        accuracies.append(acc)
        if k % 32 == 0 or k == args.n_mels - 1:
            print(f"  cutoff bin {k:3d}  ({freqs[k]:7.1f} Hz)  →  LDA acc = {acc:.4f}")

    accuracies = np.array(accuracies)

    # -------------------------------------------------------------------------
    # Save text summary
    # -------------------------------------------------------------------------
    tag = f"{args.model}_{args.sr}"
    txt_path = os.path.join(args.results_dir, f"partial_band_accuracy_{tag}.txt")
    with open(txt_path, 'w') as f:
        f.write(f"Partial-band LDA accuracy — {args.model}  SR={args.sr} Hz\n")
        f.write(f"LR Nyquist: {lr_nyquist:.0f} Hz   HR Nyquist: {nyquist:.0f} Hz\n")
        f.write(f"Samples — real: {n_real}, fake: {n_fake}\n\n")
        f.write(f"{'Cutoff bin':>12}  {'Cutoff Hz':>12}  {'LDA accuracy':>14}\n")
        f.write("-" * 44 + "\n")
        for k, hz, acc in zip(cutoff_bins, cutoff_freqs, accuracies):
            f.write(f"{k:12d}  {hz:12.1f}  {acc:14.4f}\n")
    print(f"Saved text → {txt_path}")

    # -------------------------------------------------------------------------
    # Plot: accuracy vs cutoff frequency
    # -------------------------------------------------------------------------
    font = 16
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(cutoff_freqs, accuracies, color='steelblue', linewidth=1.5)
    ax.axhline(0.5, color='grey',  linestyle='--', linewidth=1, label='Chance (0.5)')
    ax.axhline(1.0, color='red',   linestyle='--', linewidth=1, label='Perfect (1.0)')
    ax.axvline(lr_nyquist, color='orange', linestyle=':', linewidth=1.5,
               label=f'LR Nyquist ({lr_nyquist:.0f} Hz)')
    ax.axvline(nyquist,    color='black',  linestyle=':', linewidth=1.5,
               label=f'HR Nyquist ({nyquist:.0f} Hz)')

    # Mark where accuracy first crosses key thresholds
    thresholds = [0.6, 0.7, 0.8, 0.9, 1.0]
    y_offsets = [-55, -40, -25, -10, 10]
    for thresh, y_off in zip(thresholds, y_offsets):
        crossing = np.argmax(accuracies >= thresh)
        if accuracies[crossing] < thresh:
            continue  # never reaches this threshold
        hz = cutoff_freqs[crossing]
        acc_at = accuracies[crossing]
        ax.scatter([hz], [acc_at], zorder=5, s=50, color='darkred')
        ax.annotate(f'acc≥{thresh:.1f}  @  {hz/1000:.2f} kHz',
                    xy=(hz, acc_at),
                    xytext=(-80, y_off),
                    textcoords='offset points',
                    fontsize=font - 3, color='darkred',
                    arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))

    step = 2000 if args.sr == 16000 else 4000
    xticks = list(range(0, int(nyquist) + 1, step))
    if xticks[-1] != int(nyquist):
        xticks.append(int(nyquist))
    ax.set_xlim(0, nyquist)
    ax.set_xticks(xticks)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f'{int(x)}' if x < 1000 else f'{int(x)//1000}k'))
    ax.set_xlabel("Band cutoff frequency (Hz)\n[include all mel bins up to this frequency]",
                  fontsize=font)
    ax.set_ylabel("LDA accuracy", fontsize=font)
    ax.set_title(f"Partial-band classification accuracy — {args.model.upper()}  {args.sr//1000} kHz",
                 fontsize=font + 1, fontweight='bold')
    ax.set_ylim(0.35, 1.05)
    ax.tick_params(labelsize=font - 2)
    ax.legend(fontsize=font - 2, loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = os.path.join(args.results_dir, f"partial_band_accuracy_{tag}.pdf")
    plt.savefig(pdf_path, dpi=150, bbox_inches='tight')
    plt.savefig(pdf_path.replace('.pdf', '.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot  → {pdf_path}")


if __name__ == '__main__':
    main()
