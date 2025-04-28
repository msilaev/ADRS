#!/bin/bash

# -------------------
# plot learning curves
# -------------------
#SR = 48000

python3 plt_loss_thesis.py \
        --r 3 \
        --sr 48000 \
        --model gan \
        --speaker single

python3 plt_loss_thesis.py \
        --r 3 \
        --sr 48000 \
        --model audiounet \
        --speaker single

python3 plt_loss_thesis.py \
        --r 3 \
        --sr 48000 \
        --model gan_multispeaker \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 3 \
        --model audiounet_multispeaker \
        --sr 48000 \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 3 \
        --sr 48000 \
        --model gan_alt_5_multispeaker \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 3 \
        --sr 48000 \
        --model gan_alt_3_multispeaker \
        --speaker multi

######################################
#SR = 16000
python3 plt_loss_thesis.py \
        --r 4 \
        --model gan \
        --sr 16000 \
        --speaker single

python3 plt_loss_thesis.py \
        --r 4 \
        --model gan_multispeaker \
        --sr 16000 \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 4 \
        --model audiounet \
        --sr 16000 \
        --speaker single

python3 plt_loss_thesis.py \
        --r 4 \
        --model audiounet_multispeaker \
        --sr 16000 \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 4 \
        --model gan_alt_5_multispeaker \
        --sr 16000 \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 4 \
        --model gan_alt_3_multispeaker \
        --sr 16000 \
        --speaker multi

python3 plt_loss_thesis.py \
        --r 4 \
        --model gen_dec \
        --sr 16000 \
        --speaker multi

# -------------------
# plot learning curves comparison
# -------------------
python3 plt_loss_compare_thesis.py \
        --r 4 \
        --speaker single \
        --sr 16000

python3 plt_loss_compare_thesis.py \
        --r 3 \
        --speaker single \
        --sr 48000

python3 plt_loss_compare_thesis.py \
        --r 4 \
        --speaker multi \
        --sr 16000

python3 plt_loss_compare_thesis.py \
        --r 3 \
        --speaker multi \
        --sr 48000