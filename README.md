# Core code for the Work "Efficient Mask Learning for Language Model Fine-Tuning"



##  Abstract

Parameter-efficient fine-tuning (PEFT) of pre-trained language mod-
els (PLMs) has shown promising results by updating significantly
fewer parameters than full fine-tuning. Masking-based fine-tuning
is one type of the PEFT method, by freezing the majority of the
model parameters during fine-tuning. Existing masking-based fine-
tuning methods either need to manually select the trainable param-
eters, or perform mask learning to adaptively select the trainable
parameters with high memory and computation cost. To avoid man-
ual parameter selection and the high cost of mask learning, this
paper proposes Lo w- Rank based Efficient M ask L earning (LoReML).
Different from previous methods, LoReML learns the mask matrix
for the model parameters by low-rank decomposition with a small
ratio of new parameters. After mask learning, LoReML uses the
scaled intermediate results in mask learning as warm start ini-
tialization, then freezes the masked parameters accordingly, and
fine-tunes the PLM. Moreover, LoReML exploits data sparsity to
enhance the efficiency in the masking-based fine-tuning further. We
analyze the memory and computation cost of LoReML, and experi-
mentally demonstrate the effectiveness of LoReML on the GLUE
benchmark and various generation tasks using different backbones
(RoBERTa, OPT, LLaMA and DeepSeek-MoE). The results show that
LoReML outperforms the existing methods on the GLUE bench-
mark and the generation tasks by up to 3.5% in predictive accuracy,
while reducing memory consumption and improving computation
efficiency compared with the existing mask learning methods.

## Hyperparameters

Please refer to appendix.pdf for the hyperparameters used in our experiments.

## Qucik Start

We will add more instructions to run our code later.