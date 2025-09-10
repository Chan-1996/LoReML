CUDA_VISIBLE_DEVICES=0 accelerate launch -m lm_eval \
    --model hf \
    --model_args pretrained=xxx \
    --tasks mmlu \
    --output_path mmlu_results \
    --num_fewshot 5 \
    --batch_size auto \
    --cache_requests true