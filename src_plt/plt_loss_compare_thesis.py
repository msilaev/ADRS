import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=4)
    #parser.add_argument('--model')
    parser.add_argument('--speaker')
    parser.add_argument('--sr')

    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()

    if args.sr == '16000' and args.speaker == "single":

        audiounet_logs_dir_name = "/singlespeaker/sr16000/logsAudiounet/"
        gan_logs_dir_name = "/singlespeaker/sr16000/logsGAN/"
        sg_logs_dir_name = "/singlespeaker/sr16000/logsGenDecoupled/"

        logs_dir_name_save = "/compare16/singlespeaker/"

        file_name_logs_audiounet = "../logs" + audiounet_logs_dir_name + "singlespeaker.r_" + \
                                   str(args.r) + ".gan.b16.sr_16000_loss_val_gan"

        file_name_logs_gan = "../logs" + gan_logs_dir_name + "singlespeaker.r_" + \
                                           str(args.r) + ".gan.b16.sr_16000_loss_val_gan"

        file_name_logs_sg = "../logs" + sg_logs_dir_name + "singlespeaker.r_" + \
                             str(args.r) + ".gan.b16.sr_16000_loss_val_gan"

        file_name = "singlespeaker.r_" + str(args.r) + "." + \
                    "generator.b16.sr_" + str(args.sr) + "_SNR_loss"

        figure_file_name_SNR = "../results/learning_curves/" + logs_dir_name_save + file_name

        file_name = "singlespeaker.r_" + str(args.r) + "." + \
                    "generator.b16.sr_" + str(args.sr) + "_LSD_loss"

        figure_file_name_LSD = "../results/learning_curves/" + logs_dir_name_save + file_name

########################################################
    if args.sr == '48000' and args.speaker == "single" :

        audiounet_logs_dir_name = "/singlespeaker/sr48000/logsAudiounet/"
        gan_logs_dir_name = "/singlespeaker/sr48000/logsGAN/"
        sg_logs_dir_name = "/singlespeaker/sr48000/logsGenDecoupled/"

        logs_dir_name_save = "/compare48/singlespeaker/"

        file_name_logs_audiounet = "../logs" + audiounet_logs_dir_name + "singlespeaker.r_" + \
                                       str(args.r) + ".gan.b16.sr_48000_loss_val_gan"

        file_name_logs_gan = "../logs" + gan_logs_dir_name + "singlespeaker.r_" + \
                                 str(args.r) + ".gan.b16.sr_48000_loss_val_gan"

        file_name_logs_sg = "../logs" + sg_logs_dir_name + "singlespeaker.r_" + \
                                str(args.r) + ".gan.b16.sr_48000_loss_val_gan"

        file_name = "singlespeaker.r_" + str(args.r) + "." + \
                        "generator.b16.sr_" + str(args.sr) + "_SNR_loss"

        figure_file_name_SNR = "../results/learning_curves/" + logs_dir_name_save + file_name

        file_name = "singlespeaker.r_" + str(args.r) + "." + \
                        "generator.b16.sr_" + str(args.sr) + "_LSD_loss"

        figure_file_name_LSD = "../results/learning_curves/" + logs_dir_name_save + file_name

########################################################
    if args.sr == '48000' and args.speaker == "multi":

        audiounet_logs_dir_name = "/multispeaker/sr48000/logsAudiounet/"
        gan_logs_dir_name = "/multispeaker/sr48000/logsGAN_Alt5/"
        sg_logs_dir_name = "/multispeaker/sr48000/logsGenDecoupled/"

        logs_dir_name_save = "/compare48/multispeaker/"

        file_name_logs_audiounet = "../logs" + audiounet_logs_dir_name + "multispeaker.r_" + \
                                       str(args.r) + ".gan.b128.sr_48000_loss_val_gan"

        file_name_logs_gan = "../logs" + gan_logs_dir_name + "multispeaker.r_" + \
                                 str(args.r) + ".gan.b128.sr_48000_loss_val_gan"

        file_name_logs_sg = "../logs" + sg_logs_dir_name + "multispeaker.r_" + \
                                str(args.r) + ".gan.b128.sr_48000_loss_val_gan"

        file_name = "multispeaker.r_" + str(args.r) + "." + \
                        "generator.b128.sr_" + str(args.sr) + "_SNR_loss"

        figure_file_name_SNR = "../results/learning_curves/" + logs_dir_name_save + file_name

        file_name = "multispeaker.r_" + str(args.r) + "." + \
                        "generator.b128.sr_" + str(args.sr) + "_LSD_loss"

        figure_file_name_LSD = "../results/learning_curves/" + logs_dir_name_save + file_name

########################################################
    if args.sr == '16000' and args.speaker == "multi":

        audiounet_logs_dir_name = "/multispeaker/sr16000/logsAudiounet/"
        gan_logs_dir_name = "/multispeaker/sr16000/logsGAN_Alt3/"
        sg_logs_dir_name = "/multispeaker/sr16000/logsGenDecoupled/"

        logs_dir_name_save = "/compare16/multispeaker/"

        file_name_logs_audiounet = "../logs" + audiounet_logs_dir_name + "multispeaker.r_" + \
                                       str(args.r) + ".gan.b128.sr_16000_loss_val_gan"

        file_name_logs_gan = "../logs" + gan_logs_dir_name + "multispeaker.r_" + \
                                 str(args.r) + ".gan.b128.sr_16000_loss_val_gan"

        file_name_logs_sg = "../logs" + sg_logs_dir_name + "multispeaker.r_" + \
                                str(args.r) + ".gan.b128.sr_16000_loss_val_gan"

        file_name = "multispeaker.r_" + str(args.r) + "." + \
                        "generator.b128.sr_" + str(args.sr) + "_SNR_loss"

        figure_file_name_SNR = "../results/learning_curves/" + logs_dir_name_save + file_name

        file_name = "multispeaker.r_" + str(args.r) + "." + \
                        "generator.b128.sr_" + str(args.sr) + "_LSD_loss"

        figure_file_name_LSD = "../results/learning_curves/" + logs_dir_name_save + file_name


    plt_fig(args, file_name_logs_audiounet, file_name_logs_gan, file_name_logs_sg,
                figure_file_name_SNR,  figure_file_name_LSD)

def plt_fig(args, file_name_logs_audiounet, file_name_logs_gan, file_name_logs_sg,
                figure_file_name_SNR,  figure_file_name_LSD):

        snr_audiounet = []
        lsd_audiounet = []

        snr_gan = []
        lsd_gan = []

        snr_sg = []
        lsd_sg = []

        with  open(file_name_logs_audiounet + ".txt", "r") as f:

            for line in f:
                x = line.strip().split(",")

                snr_audiounet.append(float(x[8]))
                lsd_audiounet.append(float(x[7]))

        with  open(file_name_logs_gan + ".txt", "r") as f:

            for line in f:
                x = line.strip().split(",")

                snr_gan.append(float(x[8]))
                lsd_gan.append(float(x[7]))

        with  open(file_name_logs_sg + ".txt", "r") as f:

            for line in f:
                x = line.strip().split(",")

                snr_sg.append(float(x[8]))
                lsd_sg.append(float(x[7]))

        snr_audiounet = np.array(snr_audiounet)
        lsd_audiounet = np.array(lsd_audiounet)

        snr_sg = np.array(snr_sg)
        lsd_sg = np.array(lsd_sg)

        snr_gan = np.array(snr_gan)
        lsd_gan = np.array(lsd_gan)
        ind_gan = np.arange(len(snr_gan))
        ind_sg = np.arange(len(snr_sg))
        ind_audiounet = np.arange(len(snr_audiounet))

        # Set font sizes globally
        plt.rcParams.update({'font.size': 20})  # General font size
        plt.rcParams.update({'axes.titlesize': 20})  # Title font size
        plt.rcParams.update({'axes.labelsize': 20})  # X and Y label font size
        plt.rcParams.update({'legend.fontsize': 20})  # Legend font size
        plt.rcParams.update({'xtick.labelsize': 20})  # X tick label font size
        plt.rcParams.update({'ytick.labelsize': 20})  # Y tick label font size

        x_label = "Epoch, $10^2$"

        ################### SNR #####################
        plt.figure(figsize=(5, 5))

        plt.plot(ind_gan/100, snr_gan, label = 'GAN', marker="o")
        plt.plot(ind_sg/100, snr_sg, label =  'Gen dec', marker="+")
        plt.plot(ind_audiounet/100, snr_audiounet, label =  'Audiounet')

        plt.xlabel(x_label)
        plt.ylabel("SNR")
        plt.ylim([0, 35])
        plt.xlim([0, 5])
        plt.xticks([0, 1, 2, 3, 4, 5])

        plt.legend()
        plt.grid()
        plt.tight_layout(pad=0)

        plt.savefig(figure_file_name_SNR + ".png", format='png')

        ################### LSD #####################
        plt.figure(figsize=(5, 5))

        plt.plot(ind_gan/100, lsd_gan, label = 'GAN', marker="o")
        plt.plot(ind_sg/100, lsd_sg, label =  'Gen dec', marker="+")
        plt.plot(ind_audiounet/100, lsd_audiounet, label =  'Audiounet')

        plt.xlabel(x_label)
        plt.ylabel("LSD")
        plt.ylim([0, 8])
        plt.xticks([0, 1, 2, 3, 4, 5])

        plt.xlim([0, 5])

        plt.legend()
        plt.grid()
        plt.tight_layout(pad=0)

        plt.savefig(figure_file_name_LSD + ".png", format='png')

if __name__ == '__main__':
  main()





