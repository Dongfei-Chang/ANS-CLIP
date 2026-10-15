#!/bin/bash

cd ..

# -------------------- 1. 通用配置 --------------------
DATA=
TRAINER=ANS
# WEIGHT=9.0
WEIGHT=8.0
WEIGHT_V=6.0
CROSS_YAML=cross.yaml          # 统一 config
CTP=end
NCTX=5
NCTX_V=3
SHOTS=16
CSC=False
FOLDER=ans_cross
IND=0
checkFlag=1      # 0 表示 False，1 表示 True

export CUDA_VISIBLE_DEVICES=1
# -----------------------------------------------------

TARGET_DATASETS=(
    "eurosat" "dtd" "fgvc_aircraft" "ucf101" "oxford_pets"
    "stanford_cars" "oxford_flowers" "caltech101" "food101" "sun397"
    "imagenetv2" "imagenet_sketch" "imagenet_a" "imagenet_r"
)
SOURCE_DATASET="imagenet"

start_time=$(date +%s)
start_mem=$(free -m | awk '/Mem:/ {print $3}')
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training on ${SOURCE_DATASET} start | Memory used: ${start_mem} MB" >> ./imagenet_cross_domain.log

# ******************** 训练：ImageNet（串行） ********************
for SEED in 1 2 3; do
    GPU1=0; GPU2=0
    DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/cross_domain/train_${SOURCE_DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CROSS_YAML%.yaml}/seed${SEED}
    if [ -d "$DIR" ]; then
        echo "Training results exist in ${DIR}. Skipping..."
    else
        echo "Training on ${SOURCE_DATASET} (Seed ${SEED}), saving to ${DIR}"
        python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${SOURCE_DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CROSS_YAML} \
            --output-dir ${DIR} \
            --device_img cuda:${GPU1} \
            --device_txt cuda:${GPU2} \
            --checkFlag ${checkFlag} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.N_CTX_V ${NCTX_V} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.W ${WEIGHT} \
            TRAINER.COOP.L_IND ${IND} \
            TRAINER.COOP.W_V ${WEIGHT_V} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS}
    fi
done

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Training on ${SOURCE_DATASET} completed." >> ./imagenet_cross_domain.log

# ******************** 测试：跨域 / 泛化（串行） ********************
for TARGET_DATASET in "${TARGET_DATASETS[@]}"; do
    start_time_test=$(date +%s)
    start_mem_test=$(free -m | awk '/Mem:/ {print $3}')

    # 统一用 3-epoch 权重（ImageNet 只跑了3 epoch）
    LOADEP=5
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing ${SOURCE_DATASET}->${TARGET_DATASET} start | Config: ${CROSS_YAML} | Memory: ${start_mem_test} MB" >> ./imagenet_cross_domain.log

    TEST_JOBS=""
    for SEED in 1 2 3; do
        GPU1=0
        GPU2=0
        # if [ "$SEED" -lt 3 ]; then
        # GPU1=$((SEED-1))
        # GPU2=$((SEED-1))
        # else
        #     GPU1=$((SEED-2))
        #     GPU2=$((SEED-2))
        # fi
        COMMON_DIR=shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CROSS_YAML%.yaml}/seed${SEED}
        MODEL_DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/cross_domain/train_${SOURCE_DATASET}/${COMMON_DIR}
        DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/cross_domain/test_${SOURCE_DATASET}_to_${TARGET_DATASET}/${COMMON_DIR}

        if [ -d "$DIR" ]; then
            echo "Test results exist in ${DIR}. Skipping..."
        else
            echo "Testing ${SOURCE_DATASET}->${TARGET_DATASET} (Seed ${SEED}), saving to ${DIR}"
            python train.py \
                --root ${DATA} \
                --seed ${SEED} \
                --trainer ${TRAINER} \
                --dataset-config-file configs/datasets/${TARGET_DATASET}.yaml \
                --config-file configs/trainers/${TRAINER}/${CROSS_YAML} \
                --output-dir ${DIR} \
                --model-dir ${MODEL_DIR} \
                --load-epoch ${LOADEP} \
                --eval-only \
                --device_img cuda:${GPU1} \
                --device_txt cuda:${GPU2} \
                TRAINER.COOP.N_CTX ${NCTX} \
                TRAINER.COOP.N_CTX_V ${NCTX_V} \
                TRAINER.COOP.L_IND ${IND} \
                TRAINER.COOP.CSC ${CSC} \
                TRAINER.COOP.W ${WEIGHT} \
                TRAINER.COOP.W_V ${WEIGHT_V} \
                TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
                DATASET.NUM_SHOTS ${SHOTS} &
            TEST_JOBS="$TEST_JOBS $!"
        fi
    done
   
    wait ${TEST_JOBS}

    end_time_test=$(date +%s)
    elapsed_test=$((end_time_test - start_time_test))
    mins_test=$((elapsed_test / 60))
    secs_test=$((elapsed_test % 60))
    mem_diff_test=$(free -m | awk '/Mem:/ {print $3}' | awk -v m0=$start_mem_test '{print $1-m0}')
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Testing ${SOURCE_DATASET}->${TARGET_DATASET} done | Elapsed: ${mins_test}m${secs_test}s | Memory Δ: ${mem_diff_test} MB" >> ./imagenet_cross_domain.log
done

end_time=$(date +%s)
elapsed=$((end_time - start_time))
echo "[$(date '+%Y-%m-%d %H:%M:%S')] All experiments completed. Total time: $((elapsed/3600))h$((elapsed%3600/60))m" >> ./imagenet_cross_domain.log