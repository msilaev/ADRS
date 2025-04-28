#!/bin/bash

pip install hifi-gan-bwe 

make run_evaluation_singlespeaker
make run_evaluation_multispeaker
make run_evaluation_multispeaker_hifigan



