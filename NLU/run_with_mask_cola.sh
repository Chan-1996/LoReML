export PYTHONPATH=xxx
TASK_NAME=cola
WARMUP_RATIO=0.06
LR=2e-4
BATCH=32
EPOCHS=30
KEEP_RATIO=0.005
TOPK_LEVEL=group_largest
MASK_PATH=xxx/lora.bin
output_dir=outputs/${TASK_NAME}/roberta-large/learning_with_mask_keep${KEEP_RATIO}_lr${LR}_e${EPOCHS}_bz${BATCH}_wp${WARMUP_RATIO}_length128_${TOPK_LEVEL}_seed43/
mkdir -p ${output_dir}
log=${output_dir}/training.log

CUDA_VISIBLE_DEVICES=7 python3 run_glue_with_mask.py \
  --model_name_or_path roberta-large \
  --task_name ${TASK_NAME} \
  --max_length 128 \
  --pad_to_max_length \
  --weight_decay 0.1 \
  --topk_level ${TOPK_LEVEL} \
  --per_device_train_batch_size ${BATCH} \
  --learning_rate ${LR} \
  --mask_param_path ${MASK_PATH} \
  --use_mask_param_to_init_model \
  --mask_weight_scaling 1 \
  --keep_ratio ${KEEP_RATIO} \
  --warmup_ratio ${WARMUP_RATIO} \
  --num_train_epochs ${EPOCHS} \
  --seed 43 \
  --output_dir ${output_dir} > ${log} 2>&1
