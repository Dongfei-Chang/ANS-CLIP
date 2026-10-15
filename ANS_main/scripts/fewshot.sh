#!/bin/bash
cd ..

# -------------------- 通用配置 --------------------
DATA=
TRAINER=ANS
WEIGHT=8.0
WEIGHT_V=6.0

fewshot_YAML=few_shot.yaml      # ✅ 唯一 config 入口


CTP=end
NCTX=4
NCTX_V=4
CSC=False
FOLDER=output_fewshot

IND=0
checkFlag=0
protoFlag=1
EPOCH_b32=65               # b32ep50 数据集
EPOCH_b128=6               # 128ep3  数据集
GPU1=0
GPU2=0
export CUDA_VISIBLE_DEVICES=4

for SHOTS in 4 16 8 2 1; do

    protoFlag=1
    if [ $SHOTS -lt 4 ]; then
        protoFlag=0
    fi
    LOG="few_shot_time_shots${SHOTS}.log"
    echo "[$(date '+%F %T')] =====  START  SHOTS=$SHOTS  =====" >> $LOG

    # for ds in imagenet ; do
    for ds in imagenet sun397  ; do
        checkFlag=1
        start_time=$(date +%s)
        start_mem=$(free -m | awk '/Mem:/ {print $3}')
        echo "[$(date '+%F %T')] $ds  start | shots=$SHOTS | config=$fewshot_YAML | mem=${start_mem}MB" >> $LOG

        for SEED in 1 2 3; do
            GPU1=0
            GPU2=0
            DIR=${FOLDER}/shots${SHOTS}/${ds}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${fewshot_YAML%.yaml}/seed${SEED}
            [ -d "$DIR" ] && { echo "Skip existing $DIR"; continue; }

            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${ds}.yaml \
                --config-file configs/trainers/${TRAINER}/${fewshot_YAML} \
                --output-dir ${DIR} \
                --device_img cuda:${GPU1} \
                --device_txt cuda:${GPU2} \
                --checkFlag ${checkFlag} \
                --protoFlag ${protoFlag} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.N_CTX_V ${NCTX_V} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.W ${WEIGHT} \
                TRAINER.COOP.L_IND ${IND} \
                OPTIM.MAX_EPOCH ${EPOCH_b128} \
                TRAINER.COOP.W_V ${WEIGHT_V} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}
        done

        end_time=$(date +%s)
        elapsed=$((end_time-start_time))
        mem_now=$(free -m | awk '/Mem:/ {print $3}')
        mem_delta=$((mem_now-start_mem))
        echo "[$(date '+%F %T')] $ds  done | shots=$SHOTS | elapsed: $((elapsed/60))m$((elapsed%60))s | memΔ: ${mem_delta}MB" >> $LOG
    done

    for ds in eurosat dtd  fgvc_aircraft ucf101 oxford_pets stanford_cars oxford_flowers caltech101 food101; do
        checkFlag=0
        GPU1=0
        GPU2=0
        start_time=$(date +%s)
        start_mem=$(free -m | awk '/Mem:/ {print $3}')
        echo "[$(date '+%F %T')] $ds  start | shots=$SHOTS | config=$fewshot_YAML | mem=${start_mem}MB" >> $LOG

        # 将并行的seed循环改为串行
        for SEED in 1 2 3; do

            DIR=${FOLDER}/shots${SHOTS}/${ds}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${fewshot_YAML%.yaml}/seed${SEED}
            [ -d "$DIR" ] && { echo "Skip existing $DIR"; continue; }

            echo "[$(date '+%F %T')] Running $ds seed $SEED on GPU $GPU1,$GPU2" >> $LOG
            
            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${ds}.yaml \
                --config-file configs/trainers/${TRAINER}/${fewshot_YAML} \
                --output-dir ${DIR} \
                --device_img cuda:${GPU1} \
                --device_txt cuda:${GPU2} \
                --protoFlag ${protoFlag} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.N_CTX_V ${NCTX_V} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.W ${WEIGHT} \
                TRAINER.COOP.L_IND ${IND} \
                OPTIM.MAX_EPOCH ${EPOCH_b32} \
                TRAINER.COOP.W_V ${WEIGHT_V} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS}

            echo "[$(date '+%F %T')] Finished $ds seed $SEED" >> $LOG
        done

        end_time=$(date +%s)
        elapsed=$((end_time-start_time))
        mem_now=$(free -m | awk '/Mem:/ {print $3}')
        mem_delta=$((mem_now-start_mem))
        echo "[$(date '+%F %T')] $ds  done | shots=$SHOTS | elapsed: $((elapsed/60))m$((elapsed%60))s | memΔ: ${mem_delta}MB" >> $LOG
    done

    echo "[$(date '+%F %T')] =====  FINISH SHOTS=$SHOTS  =====" >> $LOG
done