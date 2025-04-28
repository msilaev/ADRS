# parser
import matplotlib.pyplot as plt
import numpy as np
import argparse
import pandas as pd
# -------------------
def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=4)
    parser.add_argument('--model')
    parser.add_argument('--speaker')
    parser.add_argument('--sr')

    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()

    if (args.sr == '16000' and args.speaker == "multi"):

        fig_save_dir = "../results/histograms16/"

        file_name_gan = "../results/MOS/scores_16_gan.csv"
        file_name_gan_5 = "../results/MOS/scores_16_gan_alt_5.csv"
        file_name_gan_3 = "../results/MOS/scores_16_gan_alt_3.csv"
        file_name_adiounet = "../results/MOS/scores_16_audiounet.csv"
        #file_name_sg = "logs" + sg_logs_dir_name + "scores_sg.csv"

        plot_histogram(file_name_gan, fig_save_dir, args.sr, file_name_gan_3, file_name_gan_5, file_name_adiounet )
        plot_skatter(file_name_gan, fig_save_dir, args.sr, file_name_adiounet)

    if (args.sr == '48000' and args.speaker == "multi"):

        fig_save_dir = "../results/histograms48/"

        file_name_gan = "../results/MOS/scores_48_gan.csv"
        file_name_adiounet = "../results/MOS/scores_48_audiounet.csv"
        file_name_gan_5 = "../results/MOS/scores_48_audiounet.csv"

        plot_histogram(file_name_gan,  fig_save_dir, args.sr,  file_name_adiounet)
        plot_skatter(file_name_gan, fig_save_dir, args.sr, file_name_gan_5, file_name_adiounet)

def plot_skatter(file_name, fig_save_dir, sr, file_name_adiounet ):

    font_size = 20

    if sr == "16000":
        title_str = "4->16 KHz"

    elif sr == "48000":
        title_str = "16->48 KHz"

    df = pd.read_csv(file_name)
     # Assuming MOS_hr and MOS_pr are pandas Series from your dataframe 'df'
    MOS_hr = df["P808_MOS hr"]
    MOS_pr = df["P808_MOS pr"]

    df_aunet = pd.read_csv(file_name_adiounet)
    MOS_pr_aunet = df_aunet["P808_MOS pr"]

    # Create the scatter plot
    fig_1, ax_1 = plt.subplots(1, 1, figsize=(5, 5))

    #plt.figure(figsize=(8, 6))  # Adjust the size as needed
    ax_1.scatter(MOS_hr, MOS_pr, color='blue', alpha=0.7, edgecolors='k')
    # Add labels and title

    # Labels
    ax_1.set_xlabel("MOS hr", fontsize=font_size)
    ax_1.set_ylabel("MOS pr", fontsize=font_size)
    ax_1.tick_params(axis='both', which='major', labelsize=font_size)

    ax_1.set_xlim([0, 5])
    ax_1.set_ylim([0, 5])
    ax_1.set_xticks([0, 1,2,3,4, 5])
    ax_1.grid()
    fig_1.tight_layout()  # Automatically adjusts subplot parameters
    fig_1.savefig(fig_save_dir + "ScatterMOS.png", format='png')


def plot_histogram(file_name_gan, fig_save_dir, sr, file_name_gan_3, file_name_gan_5, file_name_adiounet ):

    font_size = 20

    if sr == "16000":
        title_str = "4->16 KHz"

    elif sr == "48000":
        title_str = "16->48 KHz"

    df = pd.read_csv(file_name_gan)
    MOS_hr = df["P808_MOS hr"]
    MOS_pr = df["P808_MOS pr"]

    BAK_hr = df["BAK hr"]
    BAK_pr = df["BAK pr"]

    SIG_hr = df["SIG hr"]
    SIG_pr = df["SIG pr"]

    OVRL_hr = df["OVRL hr"]
    OVRL_pr = df["OVRL pr"]

    # filename	snr_pr	snr_lbr	lsd_pr	lsd_lbr	P808_MOS pr	SIG_raw pr	BAK_raw pr	OVRL pr	SIG pr	BAK pr	P808_MOS hr	SIG_raw hr
    # BAK_raw hr	OVRL hr	SIG hr	BAK hr	P808_MOS lbr	SIG_raw lbr	BAK_raw lbr	OVRL lbr	SIG lbr	BAK lbr
    SNR_pr = df["snr_pr"]
    MOS_pr_spline = df["P808_MOS lbr"]
    SNR_lbr_aunet = df["snr_lbr"]

########################################33333
    df_aunet = pd.read_csv(file_name_adiounet)
    MOS_pr_aunet = df_aunet["P808_MOS pr"]
    MOS_pr_spline = df_aunet["P808_MOS lbr"]
    SNR_pr_aunet = df_aunet["snr_pr"]
    SNR_lbr_aunet = df_aunet["snr_lbr"]

##############################################
    df_gan_3 = pd.read_csv(file_name_gan_3)
    SNR_pr_gan_3 = df_gan_3["snr_pr"]
    MOS_pr_gan_3 = df_gan_3["P808_MOS pr"]

    df_gan_5 = pd.read_csv(file_name_gan_5)
    SNR_pr_gan_5 = df_gan_5["snr_pr"]
    MOS_pr_gan_5 = df_gan_5["P808_MOS pr"]

    #############################################
    ###############
    #############################################

    fig_0, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
    fig_0.subplots_adjust(hspace=0.3)  # Adjust spacing between subplots

    # Subplot 1
    fixed_bins = np.linspace(0, 5, 51)  # 30 bins from 2 to 5
    ax_0 = ax[0]
    ax_0.hist(MOS_pr_aunet, bins=fixed_bins, alpha=1.0, color="green", label="audiounet")
    ax_0.hist(MOS_pr, bins=fixed_bins, alpha=0.8, color="magenta", label="GAN")
    ax_0.hist(MOS_hr, bins=fixed_bins, alpha=0.4, color="blue", label="hr")

    ax_0.set_xlim([2, 5])
    #ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.tick_params(axis='both', which='major', labelsize=font_size)
    ax_0.legend(fontsize=font_size)
    ax_0.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_0.set_title("Audiounet vs. HR vs. GAN", fontsize=font_size)

    # Subplot 2
    ax_1 = ax[1]
    ax_1.hist(MOS_pr_aunet, bins=fixed_bins, alpha=1.0, color="green", label="audiounet")
    ax_1.hist(MOS_pr_gan_3, bins=fixed_bins, alpha=0.8, color="magenta", label="GAN alt 3")
    ax_1.hist(MOS_hr, bins=fixed_bins, alpha=0.4, color="blue", label="hr")

    ax_1.set_ylabel("# Samples", fontsize=font_size)
    ax_1.tick_params(axis='both', which='major', labelsize=font_size)
    ax_1.legend(fontsize=font_size)
    ax_1.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_1.set_title("Audiounet vs. GAN Alt 3 vs. HR", fontsize=font_size)

    # Subplot 3
    ax_2 = ax[2]
    ax_2.hist(MOS_pr_aunet, bins=fixed_bins, alpha=1.0, color="green", label="audiounet")
    ax_2.hist(MOS_pr_gan_5, bins=fixed_bins, alpha=0.8, color="magenta", label="GAN alt 5")
    ax_2.hist(MOS_hr, bins=fixed_bins, alpha=0.4, color="blue", label="hr")

    ax_2.set_xlim([2, 5])
    ax_2.set_xlabel("MOS", fontsize=font_size)
    #ax_2.set_ylabel("# Samples", fontsize=font_size)
    ax_2.tick_params(axis='both', which='major', labelsize=font_size)
    ax_2.legend(fontsize=font_size)
    ax_2.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_2.set_title("Audiounet vs. GAN Alt 5 vs. HR", fontsize=font_size)

    # Final layout adjustment
    plt.tight_layout()
    plt.show()
    fig_0.savefig(fig_save_dir + "DNMOS_hist_1.png", format='png')

    #############################################
    ###############
    #############################################

    fig_0, ax = plt.subplots(3, 1, figsize=(10, 6), sharex=True, sharey=True)
    fig_0.subplots_adjust(hspace=0.3)  # Adjust spacing between subplots
    fixed_bins = np.linspace(0, 40, 41)

    # Subplot 1
    ax_0 = ax[0]
    ax_0.hist(SNR_pr_aunet, bins=fixed_bins, alpha=1.0, color="green", label="audiounet")
    ax_0.hist(SNR_pr, bins=fixed_bins, alpha=0.8, color="magenta", label="GAN")
    #ax_0.hist(SNR_hr, bins=30, alpha=0.4, color="blue", label="hr")

    ax_0.set_xlim([0, 30])
    #ax_0.set_ylabel("# Samples", fontsize=font_size)
    ax_0.tick_params(axis='both', which='major', labelsize=font_size)
    ax_0.legend(fontsize=font_size)
    ax_0.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_0.set_title("Audiounet vs. HR vs. GAN", fontsize=font_size)

    # Subplot 2
    ax_1 = ax[1]
    ax_1.set_xlim([0, 30])
    ax_1.hist(SNR_pr_aunet, bins=fixed_bins, alpha=1.0, color="green", label="audiounet")
    ax_1.hist(SNR_pr_gan_3, bins=fixed_bins, alpha=0.8, color="magenta", label="GAN alt 3")
    #ax_1.hist(MOS_hr, bins=30, alpha=0.4, color="blue", label="hr")

    ax_1.set_ylabel("# Samples", fontsize=font_size)
    ax_1.tick_params(axis='both', which='major', labelsize=font_size)
    ax_1.legend(fontsize=font_size)
    ax_1.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_1.set_title("Audiounet vs. GAN Alt 3 vs. HR", fontsize=font_size)

    # Subplot 3
    ax_2 = ax[2]
    ax_2.hist(SNR_pr_aunet, bins=fixed_bins, alpha=1.0, color="green", label="audiounet")
    ax_2.hist(SNR_pr_gan_5, bins=fixed_bins, alpha=0.8, color="magenta", label="GAN alt 5")
   # ax_2.hist(MOS_hr, bins=30, alpha=0.4, color="blue", label="hr")

    ax_2.set_xlim([0, 30])
    ax_2.set_xlabel("SNR", fontsize=font_size)
    #ax_2.set_ylabel("# Samples", fontsize=font_size)
    ax_2.tick_params(axis='both', which='major', labelsize=font_size)
    ax_2.legend(fontsize=font_size)
    ax_2.grid(color='gray', linestyle='--', linewidth=0.5)
    #ax_2.set_title("Audiounet vs. GAN Alt 5 vs. HR", fontsize=font_size)

    # Final layout adjustment
    plt.tight_layout()
    plt.show()

    fig_0.savefig(fig_save_dir + "SNR_hist_1.png", format='png')

if __name__ == "__main__":
    main()