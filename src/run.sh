#!/bin/bash
# SVR_NAME=NIPA
SVR_NAME=Jupiter
LABEL=abl_nobalanceloss

MODEL=EDSR
DATA_ROOT='/mnt/ssd2/frontier/datasets'

LRGB_ROOT='../../lrgb_pretrained'
LRGB_NAME='ABLATION/div2k_nobalanceloss'
LRGB_MODEL='model_latest.pt'
LRGB_N_EXPERTS=20

LABEL=${MODEL}_${LABEL}

LRGB_PATH=${LRGB_ROOT}/${LRGB_NAME}/model/${LRGB_MODEL}

CUDA_VISIBLE_DEVICES=0,1 python main.py --dir_data ${DATA_ROOT} --save ${LABEL} --model ${MODEL} --n_GPUs 2 \
    --lrgb_ckpt ${LRGB_PATH} --lrgb_n_experts ${LRGB_N_EXPERTS} # --lrgb_use_legacy


SMS_MSG=${LABEL}' @ '${SVR_NAME}' is over.'
bash send_sms.sh "${SMS_MSG}"
