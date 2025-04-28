import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.io import load_h5

def make_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--r', type=int, default=4)
    parser.add_argument('--model')
    parser.add_argument('--sr')
    parser.add_argument('--speaker')

    return parser

def plt_loss_evolution(file_name_logs_val, out_dir_name, max_it):

    g_gan_loss_train = []
    d_loss_train = []

    with  open(file_name_logs_val + ".txt", "r") as f:

        for line in f:
            x = line.strip().split(",")

            d_loss_train.append( float(x[1]) )

            g_gan_loss_train.append(float(x[5]))

    d_loss_train = np.array(d_loss_train)
    g_gan_loss_train = np.array(g_gan_loss_train)

    # Create a window of size
    window_size = 50
    window = np.ones(window_size) / window_size

    # Compute the moving average using np.convolve
    g_gan_loss_train = np.convolve(g_gan_loss_train, window, mode='valid')
    d_loss_train = np.convolve(d_loss_train, window, mode='valid')

    # Set font sizes globally
    plt.figure(figsize=(5, 5))

    plt.rcParams.update({'font.size': 25})  # General font size
    plt.rcParams.update({'axes.titlesize': 25})  # Title font size
    plt.rcParams.update({'axes.labelsize': 25})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 25})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 25})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 25})  # Y tick label font size

    x_label = "Iteration, $10^3$"
    plt.ylabel("<Adv Loss>$_{50}$")

    plt.plot(np.arange( g_gan_loss_train.shape[0])/1000, g_gan_loss_train, label='Gen', marker = "o")
    plt.plot(np.arange( d_loss_train.shape[0])/1000, d_loss_train, label='Disc', marker = "+")

    plt.xlabel(x_label)
    plt.ylim([0,2])
    plt.xlim([0, max_it])
    plt.xticks([ i for i in range(0,max_it+100, max_it//3) ])

    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0)

    figure_file_name = "../results/learning_curves/" + out_dir_name + "evolution.png"

    #print(figure_file_name)
    #plt.show()

    plt.savefig(figure_file_name, format='png')

def plt_loss_epoch(file_name_logs_val, file_name_logs_train, out_dir_name, args):

    g_loss_val = []
    g_gan_loss_val = []
    d_loss_val = []
    mse_loss_val = []
    snr_val =[]
    lsd_val = []

    g_loss_train = []
    g_gan_loss_train = []
    d_loss_train = []
    mse_loss_train = []

    snr_train =[]
    lsd_train = []

    with  open(file_name_logs_val + ".txt", "r") as f:

        for line in f:
            x = line.strip().split(",")

            d_loss_val.append(float(x[1]) + float(x[2]))

            g_loss_val.append(float(x[3]))
            g_gan_loss_val.append(float(x[4]))
            mse_loss_val.append(float(x[5]))

            snr_val.append(float(x[8]))
            lsd_val.append(float(x[7]))

    with  open(file_name_logs_train + ".txt", "r") as f:

        for line in f:
            x = line.strip().split(",")

            d_loss_train.append(float(x[1]) + float(x[2]))

            g_loss_train.append(float(x[3]))
            g_gan_loss_train.append(float(x[4]))
            mse_loss_train.append(float(x[5]))

            snr_train.append(float(x[8]))
            lsd_train.append(float(x[7]))

    g_gan_loss_val = np.array(g_gan_loss_val)
    mse_loss_val = np.array(mse_loss_val)
    d_loss_val = np.array(d_loss_val)
    lsd_val = np.array(lsd_val)
    snr_val = np.array(snr_val)

    mse_loss_train = np.array(mse_loss_train)

    lsd_train = np.array(lsd_train)
    snr_train = np.array(snr_train)

    plt.rcParams.update({'font.size': 25})  # General font size
    plt.rcParams.update({'axes.titlesize': 25})  # Title font size
    plt.rcParams.update({'axes.labelsize': 25})  # X and Y label font size
    plt.rcParams.update({'legend.fontsize': 25})  # Legend font size
    plt.rcParams.update({'xtick.labelsize': 25})  # X tick label font size
    plt.rcParams.update({'ytick.labelsize': 25})  # Y tick label font size

    ################### SNR #####################
    x_label = "Epoch, $10^2$"
    plt.xticks([0, 1, 2, 3, 4, 5])
    plt.xlim([0, 5.0])

    if args.speaker == "single":
        file_name = "singlespeaker.r_" + str(args.r) + "." + args.model + \
                        ".generator.b16.sr_" + str(args.sr) + "_SNR_loss"
    else:
        file_name = "multispeaker.r_" + str(args.r) + "." + args.model + \
                    ".generator.b16.sr_" + str(args.sr) + "_SNR_loss"

    plt.figure(figsize=(5, 5))

    plt.plot(np.arange( snr_val.shape[0])/100, snr_val, label ='val', marker="o")
    plt.plot(np.arange( snr_train.shape[0])/100, snr_train, label = 'train', marker="+")
    plt.xlabel(x_label)

    plt.ylabel("SNR")
    plt.ylim([0,30])
    plt.yticks([0, 10, 20, 30])
    plt.xticks([0, 1, 2, 3, 4, 5])
    if args.speaker == "multi":
        plt.xlim([0, 5.0])
    elif args.speaker == "single":
        plt.xlim([0, 5.0])
    #plt.title("r=" + str(args.r) + ", " + args.model + ", " + \
    #                  "sr=" + str(int(args.sr)//1000) + " KHz")
    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0)

    figure_file_name = "../results/learning_curves/" + out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')

    ################### LSD #####################
    if args.speaker == "single":
        file_name = "singlespeaker.r_" + str(args.r) + "." + args.model + \
                    ".generator.b16.sr_" + str(args.sr) + "_LSD_loss"
    else:
        file_name = "multispeaker.r_" + str(args.r) + "." + args.model + \
                    ".generator.b16.sr_" + str(args.sr) + "_LSD_loss"

    plt.figure(figsize=(5, 5))

    plt.plot(np.arange( lsd_val.shape[0])/100, lsd_val, label= 'val', marker="o")
    plt.plot(np.arange( lsd_train.shape[0])/100, lsd_train, label= 'train', marker="+")
    plt.xlabel(x_label)
    plt.ylabel("LSD")
    plt.ylim([0, 8])
    plt.yticks([0,  2,  4, 6, 8 ])
    plt.xticks([0, 1, 2, 3, 4, 5])
    if args.speaker == "multi":
        plt.xlim([0, 5.0])
    elif args.speaker == "single":
        plt.xlim([0, 5.0])

    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0)

    figure_file_name = "../results/learning_curves/" + out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')

    ############## MSE #####################
    if args.speaker == "single":
        file_name = "singlespeaker.r_" + str(args.r) + "." + args.model + \
                        ".generator.b16.sr_" + str(args.sr) + "_MSE_loss"

    else :
        file_name = "multispeaker.r_" + str(args.r) + "." + args.model + \
                        ".generator.b16.sr_" + str(args.sr) + "_MSE_loss"

    plt.figure(figsize=(6, 5))

    plt.plot(np.arange( mse_loss_val.shape[0])/100, mse_loss_val , label='val', marker="o")
    plt.plot(np.arange( mse_loss_train.shape[0])/100, mse_loss_train, label='train', marker="+")
    plt.xlabel(x_label)
    plt.ylabel("MSE")
    plt.ylim([0, 0.0001])

    plt.legend()
    plt.xticks([0, 1, 2, 3, 4, 5])

    plt.xlim([0, 5.0])
    #plt.title("r=" + str(args.r) + ", " + args.model + ", " + \
    #          "sr=" + str(int(args.sr)//1000) + " KHz")

    plt.grid()
    plt.tight_layout(pad=0)

    figure_file_name = "../results/learning_curves/" + out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')

    ########### Gen Adv ########################
    if args.speaker == "single":
        file_name = "singlespeaker.r_" + str(args.r) + "." + args.model + \
                        ".generator.b16.sr_" + str(args.sr) + "_adv_loss"
    else:
        file_name = "multispeaker.r_" + str(args.r) + "." + args.model + \
                    ".generator.b16.sr_" + str(args.sr) + "_adv_loss"

    plt.figure(figsize=(5, 5))

    plt.plot(np.arange(g_gan_loss_val.shape[0])/100, g_gan_loss_val, label='Gen', marker="o")
    plt.plot(np.arange( d_loss_val.shape[0])/100, d_loss_val, label='Disc', marker="+")

    x_label = "Epoch, $10^2$"
    plt.ylabel('Adv Loss')
    plt.xlabel(x_label)
    plt.ylim([0, 4.0])
    plt.xlim([0, 5.0])
    plt.xticks([0, 1, 2, 3, 4, 5])

    plt.legend()
    plt.grid()
    plt.tight_layout(pad=0.0)

    figure_file_name = "../results/learning_curves/" + out_dir_name + file_name
    plt.savefig(figure_file_name + ".png", format='png')


def main():

    parser = make_parser()
    args = parser.parse_args()

    if args.sr == '16000' and args.model == "gan":

        logs_dir_name = "/singlespeaker/sr16000/logsGAN/"
        out_dir_name = "/singlespeaker/sr16000/gan/"
        logs_fname_evol = logs_dir_name + "singlespeaker.r_4.gan.b16.sr_16000_loss_gan"
        max_it = 100

    elif args.sr == '16000' and args.model == "gan_multispeaker":

        logs_dir_name = "/multispeaker/sr16000/logsGAN/"
        out_dir_name = "/multispeaker/sr16000/gan/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_4.gan.b128.sr_16000_loss_gan"
        max_it = 300

    elif args.sr == '16000' and args.model == "audiounet_multispeaker":

        logs_dir_name = "/multispeaker/sr16000/logsAudiounet/"
        out_dir_name = "/multispeaker/sr16000/audiounet/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_4.gan.b128.sr_16000_loss_gan"
        max_it = 300

    elif args.sr == '16000' and args.model == "audiounet":

        logs_dir_name = "/singlespeaker/sr16000/logsAudiounet/"
        out_dir_name = "/singlespeaker/sr16000/audiounet/"
        logs_fname_evol = logs_dir_name + "singlespeaker.r_4.gan.b16.sr_16000_loss_gan"
        max_it = 300

    elif args.sr == '16000' and args.model == "gan_alt_5_multispeaker":

        logs_dir_name = "/multispeaker/sr16000/logsGAN_Alt5/"
        out_dir_name = "/multispeaker/sr16000/gan_alt_5/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_4.gan.b128.sr_16000_loss_gan"
        max_it = 300

    elif args.sr == '16000' and args.model == "gan_alt_3_multispeaker":

        logs_dir_name = "/multispeaker/sr16000/logsGAN_Alt3/"
        out_dir_name = "/multispeaker/sr16000/gan_alt_3/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_4.gan.b128.sr_16000_loss_gan"
        max_it = 300

    elif args.sr == '16000' and args.model == "gen_dec":

        logs_dir_name = "/multispeaker/sr16000/logsGenDecoupled/"
        out_dir_name = "/multispeaker/sr16000/gen_dec/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_4.gan.b128.sr_16000_loss_gan"
        max_it = 300

################################################################3
    elif args.sr == '48000' and args.r == 3 and args.model == "gan_multispeaker":

        logs_dir_name = "/multispeaker/sr48000/logsGAN/"
        out_dir_name = "/multispeaker/sr48000/gan/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_3.gan.b128.sr_48000_loss_gan"
        max_it = 300*3

    elif args.sr == '48000' and args.r == 3 and args.model == "gan":

        logs_dir_name = "/singlespeaker/sr48000/logsGAN/"
        out_dir_name = "/singlespeaker/sr48000/gan/"
        logs_fname_evol = logs_dir_name + "singlespeaker.r_3.gan.b16.sr_48000_loss_gan"
        max_it = 300

    elif args.sr == '48000' and args.r == 3 and args.model == "audiounet":

        logs_dir_name = "/singlespeaker/sr48000/logsAudiounet/"
        out_dir_name = "/singlespeaker/sr48000/audiounet/"
        logs_fname_evol = logs_dir_name + "singlespeaker.r_3.gan.b16.sr_48000_loss_gan"
        max_it = 300

    elif args.sr == '48000' and args.r == 3 and args.model == "audiounet_multispeaker":

        logs_dir_name = "/multispeaker/sr48000/logsAudiounet/"
        out_dir_name = "/multispeaker/sr48000/audiounet/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_3.gan.b128.sr_48000_loss_gan"
        max_it = 300

    elif args.sr == '48000' and args.r == 3 and args.model == "gan_alt_5_multispeaker":

        logs_dir_name = "/multispeaker/sr48000/logsGAN_Alt5/"
        out_dir_name = "/multispeaker/sr48000/gan_alt_5/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_3.gan.b128.sr_48000_loss_gan"
        max_it = 300*3

    elif args.sr == '48000' and args.r == 3 and args.model == "gan_alt_3_multispeaker":

        logs_dir_name = "/multispeaker/sr48000/logsGAN_Alt3/"
        out_dir_name = "/multispeaker/sr48000/gan_alt_3/"
        logs_fname_evol = logs_dir_name + "multispeaker.r_3.gan.b128.sr_48000_loss_gan"
        max_it = 300*3

    if args.speaker == "multi":
        batch_name = "b128"

        file_name_logs_val = "../logs" + logs_dir_name + "multispeaker.r_" + \
                             str(args.r) + ".gan." + batch_name + ".sr_" + \
                             str(args.sr) + "_loss_val_gan"

        file_name_logs_train = "../logs" + logs_dir_name + "multispeaker.r_" + \
                               str(args.r) + ".gan." + batch_name + ".sr_" + \
                               str(args.sr) + "_loss_train_gan"

    elif args.speaker == "single":
        batch_name = "b16"

        file_name_logs_val = "../logs" + logs_dir_name + "singlespeaker.r_" + \
                             str(args.r) + ".gan." + batch_name + ".sr_" + \
                             str(args.sr) + "_loss_val_gan"

        file_name_logs_train = "../logs" + logs_dir_name + "singlespeaker.r_" + \
                               str(args.r) + ".gan." + batch_name + ".sr_" + \
                               str(args.sr) + "_loss_train_gan"

    plt_loss_evolution("../logs" + logs_fname_evol, out_dir_name, max_it)
    plt_loss_epoch(file_name_logs_val, file_name_logs_train, out_dir_name, args)

if __name__ == '__main__':
  main()
