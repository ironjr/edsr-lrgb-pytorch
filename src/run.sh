#!/bin/bash
# SVR_NAME=NIPA
SVR_NAME=Jupiter
LABEL=abl_c5

MODEL=RRDB

LRGB_ROOT='../../lrgb_pretrained'
LRGB_NAME='ABLATION/div2k-class5'
LRGB_MODEL='model_latest.pt'
LRGB_N_EXPERTS=5

LABEL=${MODEL}_${LABEL}

LRGB_PATH=${LRGB_ROOT}/${LRGB_NAME}/model/${LRGB_MODEL}

CUDA_VISIBLE_DEVICES=1,2 python main.py --save ${LABEL} --model ${MODEL} --n_GPUs 2 \
    --lrgb_ckpt ${LRGB_PATH} --lrgb_n_experts ${LRGB_N_EXPERTS} --lrgb_use_legacy


SMS_MSG=${LABEL}' @ '${SVR_NAME}' is over.'
bash send_sms.sh "${SMS_MSG}"
