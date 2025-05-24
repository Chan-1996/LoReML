export PYTHONPATH=xxx
DATA_NAME=alpaca
WEIGHT_DECAY=0.01
LR=2e-4
BATCH=8
EPOCHS=3
KEEP_RATIO=0.005
TOPK_LEVEL=group_largest
SEED=43
MASK_PATH=xxx/lora.bin
output_dir=outputs/${DATA_NAME}/Ours_${KEEP_RATIO}_lr${LR}_e${EPOCHS}_bz${BATCH}_wd${WEIGHT_DECAY}_seed${SEED}
mkdir -p ${output_dir}
log=${output_dir}/training.log

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file 7b-a800-masking.yaml run_masking_alpaca.py \
  --model_name_or_path path_to_pretrained_model \
  --data_path data/alpaca_data.json \
  --max_training_seq_length 512 \
  --weight_decay ${WEIGHT_DECAY} \
  --per_device_train_batch_size ${BATCH} \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 1 \
  --learning_rate ${LR} \
  --lr_scheduler_type "linear" \
  --topk_level ${TOPK_LEVEL} \
  --keep_ratio ${KEEP_RATIO} \
  --mask_param_path ${MASK_PATH} \
  --use_mask_param_to_init_model \
  --mask_weight_scaling 0.5 \
  --num_warmup_steps 1000 \
  --num_train_epochs ${EPOCHS} \
  --seed ${SEED} \
  --output_dir ${output_dir} > ${log} 2>&1
