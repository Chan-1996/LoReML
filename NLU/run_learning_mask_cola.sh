export PYTHONPATH=xxx
TASK_NAME=cola
WEIGHT_DECAY=0
WEIGHT_RANK=4
ALPHA=8
LR=5e-5
BATCH=8
EPOCHS=20
output_dir=outputs/${TASK_NAME}/roberta-large/learning_mask_rank${WEIGHT_RANK}_alpha${ALPHA}_lr${LR}_e${EPOCHS}_bz${BATCH}_wd${WEIGHT_DECAY}_see43
mkdir -p ${output_dir}
log=${output_dir}/training.log


CUDA_VISIBLE_DEVICES=0 python3 run_glue_learning_mask.py \
  --model_name_or_path roberta-large \
  --task_name ${TASK_NAME} \
  --max_length 128 \
  --pad_to_max_length \
  --weight_rank ${WEIGHT_RANK}\
  --alpha ${ALPHA}\
  --weight_decay ${WEIGHT_DECAY} \
  --per_device_train_batch_size ${BATCH} \
  --learning_rate ${LR} \
  --lmbda 1e-8 \
  --mask_method low_rank \
  --warmup_ratio 0.06 \
  --num_train_epochs ${EPOCHS} \
  --seed 43 \
  --output_dir ${output_dir} > ${log} 2>&1
