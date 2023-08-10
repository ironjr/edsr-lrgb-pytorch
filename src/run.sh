#!/bin/bash
SVR_NAME=NIPA
LABEL=edsr

LRGB_ROOT='../../LRGB'
LRGB_NAME='in1k-'
LRGB_MODEL='model_best.pt'

LRGB_PATH=${LRGB_ROOT}/experiment/${LRGB_NAME}/model/${LRGB_MODEL}

CUDA_VISIBLE_DEVICES=0,1 python main.py --save ${LABEL} --n_GPUs 2 --lrgb_ckpt ${LRGB_PATH}


SMS_MSG=${LABEL}' @ '${SVR_NAME}' is over.'
bash send_sms.sh "${SMS_MSG}"
