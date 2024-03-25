#!/bin/bash
##!/bin/bash

############################################################
## THIS IS THE SCRIPT WITH NOISE AND THE VARIANCE CORRECTION.
############################################################

## The following is for running on theta gpu
# export http_proxy=http://proxy.tmi.alcf.anl.gov:3128
# export https_proxy=http://proxy.tmi.alcf.anl.gov:3128

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# __conda_setup="$('/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
# if [ $? -eq 0 ]; then
#     eval "$__conda_setup"
# else
#     if [ -f "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh" ]; then
#         . "/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/etc/profile.d/conda.sh"
#     else
#         export PATH="/lus/theta-fs0/software/thetagpu/conda/2021-06-26/mconda3/bin:$PATH"
#     fi
# fi
# unset __conda_setup
# # <<< conda initialize <<<
# conda activate posei

## The following is for running on JLSE
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ML__torch

python test_models.py train models -e 60000
python test_models.py list models
python test_models.py plot models
conda deactivate