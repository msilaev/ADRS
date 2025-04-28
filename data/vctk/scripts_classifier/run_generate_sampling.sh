#!/bin/bash

echo "Creating sampling..."
python ../gen_u.py \
    --file_list train-files.txt \
    --sampling_file sampling_u_01.txt \
    --sampling_len 200000 \
    --sam 0.1
