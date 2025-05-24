# Core code for the Work "Efficient Mask Learning for Language Model Fine-Tuning"


## Hyperparameters

Please refer to the appendix.pdf for the hyperparameters used in our experiments.

## Qucik Start

1. Install the dependencies
```
pip3 install -r requirement.txt
```

2. Run the experiments on the GLUE benchmark

```
cd NLU
```
(1) Mask learning: 

Modify the hyper-parameters in run_learning_mask_cola.sh, then
```
bash run_learning_mask_cola.sh
```
(2) Second-stage fine-tuning:

Modify the hyper-parameters and path of the learned mask in run_with_mask_cola.sh, then

```
bash run_with_mask_cola.sh
```
3. Run the experiments on the MMLU benchamrk
```
cd NLG
```
(1) Mask learning

Modify the hyper-parameters in run_learning_mask.sh, then
```
bash run_learning_mask.sh
```
(2) Second-stage fine-tuning:

Modify the hyper-parameters in run_masking.sh, then
```
bash run_masking.sh
```
