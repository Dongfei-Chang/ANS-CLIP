#!/bin/bash

cd ..


DATA= # path of dataset
TRAINER=ANS
WEIGHT=8.0 # weight of the textual consistency
WEIGHT_V=6.0 # weight of the visual consistency

CFG=vit_b32_ep50_ctxv1
CTP=end  # class token position (end or middle)
NCTX=8 # length of textual prompts
NCTX_V=4 # length of visual prompts
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output
IND=0 # indext of selected layers. 0: [1,2,3,4,5,6,7,8,9,10,11]

checkFlag=1
export CUDA_VISIBLE_DEVICES=0
#seed的
Seed1_GPU=0
Seed2_GPU=0
Seed3_GPU=0



#内部的
GPU1=0
GPU2=0

for NCTX in 8
do
  for NCTX_V in 4
  do
    # for DATASET in caltech101 food101 dtd  oxford_flowers  oxford_pets eurosat

    # for DATASET in  oxford_flowers 
    # for DATASET in  eurosat
    for DATASET in  eurosat dtd fgvc_aircraft ucf101 oxford_pets stanford_cars oxford_flowers caltech101 food101
    do
      start_time=$(date +%s)
      start_mem=$(free -m | awk '/Mem:/ {print $3}')
      echo "[`date '+%Y-%m-%d %H:%M:%S'`] ${DATASET} start | Memory used: ${start_mem} MB" >> ./dataset_time.log

      TRAIN_JOBS=""
      for SEED in 1 2 3
      do

        case $SEED in
            1) export CUDA_VISIBLE_DEVICES=${Seed1_GPU} ;;
            2) export CUDA_VISIBLE_DEVICES=${Seed2_GPU} ;;
            3) export CUDA_VISIBLE_DEVICES=${Seed3_GPU} ;;
        esac

        DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
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
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SPLIT train \
            DATASET.SUBSAMPLE_CLASSES base &
            TRAIN_JOBS="$TRAIN_JOBS $!"  # Collect PID of the training job
        fi
      done

      # Wait for ALL training jobs for the current dataset to complete before running test
      wait ${TRAIN_JOBS}
      echo "All training jobs for dataset ${DATASET} completed.  Starting testing."

      # Run testing jobs AFTER training is complete for all seeds
      for SEED in 1 2 3
      do

          case $SEED in
              1) export CUDA_VISIBLE_DEVICES=${Seed1_GPU} ;;
              2) export CUDA_VISIBLE_DEVICES=${Seed2_GPU} ;;
              3) export CUDA_VISIBLE_DEVICES=${Seed3_GPU} ;;
          esac

          LOADEP=50
          SUB=new
          # SUB=base
          COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
          MODEL_DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${COMMON_DIR}
          DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/test_${SUB}/${COMMON_DIR}

          if [ -d "$DIR" ]; then
              echo "Results are available in ${DIR}. Skip this job"
          else
              echo "Run this job and save the output to ${DIR}"
              python train.py \
              --root ${DATA} \
              --seed ${SEED} \
              --trainer ${TRAINER} \
              --dataset-config-file configs/datasets/${DATASET}.yaml \
              --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
              --output-dir ${DIR} \
              --model-dir ${MODEL_DIR} \
              --load-epoch ${LOADEP} \
              --eval-only \
              --device_img cuda:${GPU1} \
              --device_txt cuda:${GPU2} \
              --checkFlag ${checkFlag} \
              TRAINER.COOP.N_CTX ${NCTX} \
              TRAINER.COOP.N_CTX_V ${NCTX_V} \
              TRAINER.COOP.L_IND ${IND} \
              TRAINER.COOP.CSC ${CSC} \
              TRAINER.COOP.W ${WEIGHT} \
              TRAINER.COOP.W_V ${WEIGHT_V} \
              TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
              DATASET.NUM_SHOTS ${SHOTS} \
              DATASET.SPLIT test \
              DATASET.SUBSAMPLE_CLASSES ${SUB} &
          fi
      done
      wait #wait for all test jobs to finish before moving to the next dataset
      end_time=$(date +%s)
      end_mem=$(free -m | awk '/Mem:/ {print $3}')
      elapsed=$((end_time - start_time))
      mins=$((elapsed / 60))
      secs=$((elapsed % 60))
      mem_diff=$((end_mem - start_mem))
      echo "[`date '+%Y-%m-%d %H:%M:%S'`] ${DATASET} end, elapsed: ${mins} min ${secs} sec | Memory used: ${end_mem} MB (Δ ${mem_diff} MB)" >> ./dataset_time.log
    # echo "[`date '+%Y-%m-%d %H:%M:%S'`] ${DATASET} end, elapsed: ${mins} min ${secs} sec" >> dataset_time.log
    done
  done
done


#!/bin/bash

# custom config

DATA= # path of dataset
TRAINER=ANS
WEIGHT=8.0 # weight of the textual consistency 
WEIGHT_V=6.0 # weight of the visual consistency

CFG=vit_b32_ep50_ctxv1
CTP=end  # class token position (end or middle)
# NCTX=10 # length of textual prompts
NCTX=8 # length of textual prompts
# NCTX=6 # length of textual prompts  best-now
NCTX_V=4 # length of visual prompts
SHOTS=16  # number of shots (1, 2, 4, 8, 16)
CSC=False  # class-specific context (False or True)
FOLDER=output_b2n
IND=0 # indext of selected layers. 0: [1,2,3,4,5,6,7,8,9,10,11]

GPU1=0
GPU2=0
checkFlag=0      # 0 表示 False，1 表示 True

for NCTX in 8
do
for NCTX_V in 4
do


CFG=vit_b128_ep10_ctxv1

# for DATASET in  imagenet 
# for DATASET in imagenet 
for DATASET in imagenet sun397 
do
    start_time=$(date +%s)
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] ${DATASET} start" >> ./dataset_time.log
    for SEED in 1 2 3
    do
        DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
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
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES base
        fi
    done

    LOADEP=10
    SUB=new
    for SEED in 1 2 3
    do
        COMMON_DIR=${DATASET}/shots_${SHOTS}_${WEIGHT}/${TRAINER}/${CFG}/seed${SEED}
        MODEL_DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/train_base/${COMMON_DIR}
        DIR=${FOLDER}_${NCTX}_${IND}_${NCTX}_${NCTX_V}/base2new/test_${SUB}/${COMMON_DIR}

        if [ -d "$DIR" ]; then
            echo "Results are available in ${DIR}. Skip this job"
        else
            echo "Run this job and save the output to ${DIR}"
            python train.py \
            --root ${DATA} \
            --seed ${SEED} \
            --trainer ${TRAINER} \
            --dataset-config-file configs/datasets/${DATASET}.yaml \
            --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
            --output-dir ${DIR} \
            --model-dir ${MODEL_DIR} \
            --load-epoch ${LOADEP} \
            --eval-only \
            --device_img cuda:${GPU1} \
            --device_txt cuda:${GPU2} \
            --checkFlag ${checkFlag} \
            TRAINER.COOP.N_CTX ${NCTX} \
            TRAINER.COOP.N_CTX_V ${NCTX_V} \
            TRAINER.COOP.L_IND ${IND} \
            TRAINER.COOP.CSC ${CSC} \
            TRAINER.COOP.CLASS_TOKEN_POSITION ${CTP} \
            DATASET.NUM_SHOTS ${SHOTS} \
            DATASET.SUBSAMPLE_CLASSES ${SUB}
        fi
    done
    end_time=$(date +%s)
    elapsed=$((end_time - start_time))
    mins=$((elapsed / 60))
    secs=$((elapsed % 60))
    echo "[`date '+%Y-%m-%d %H:%M:%S'`] ${DATASET} end, elapsed: ${mins} min ${secs} sec" >> ./dataset_time.log
done
done
done
