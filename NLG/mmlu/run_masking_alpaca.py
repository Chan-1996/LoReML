import logging
import os

import transformers
from transformers import (
    MODEL_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer
)
from transformers import LlamaTokenizer
from transformers import HfArgumentParser, TrainingArguments, get_scheduler
from dataclasses import field, dataclass
import random
import numpy as np
import torch
import copy
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate.logging import get_logger
import math
from tqdm.auto import tqdm
import datasets
import json
import io
from typing import Optional, Dict, Sequence
from masking.masking_model import Masking
from spops import csr_add


# logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
#                     datefmt='%m/%d/%Y %H:%M:%S',
#                     level=logging.INFO)
# logger = logging.getLogger(__name__)
logger = get_logger(__name__)

# You should update this to your particular problem to have better documentation of `model_type`
MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)
os.environ['WANDB_MODE'] = 'disabled'

@dataclass
class OurArgs(TrainingArguments):
    model_name_or_path: str = field(default=None)
    data_path: str = field(default=None)
    data_length: int = field(default=100000)
    max_training_seq_length: int = field(default=1024)
    pad_to_max_length: bool = field(default=False)
    eval_dataset_size: float = field(
        default=0.05, metadata={"help": "0--1 means ratio, >1 means number of examples"}
    )
    seed: int = field(default=43)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    learning_rate: float = field(default=1e-5)
    lr_scheduler_type: str = field(default="linear",
                                   metadata={"help": "The scheduler type to use. Choose from (linear, cosine...)"})
    lr_decay: float = field(default=0.9)
    weight_decay: float = field(default=0)
    num_warmup_steps: int = field(default=0)
    max_train_steps: int = field(default=10000)
    num_train_epochs: int = field(default=5)
    gradient_accumulation_steps: int = field(default=1)
    output_dir: str = field(default=None)
    save_steps: int = field(default=2000)

    #masking
    mask_param_path: str = field(default=None)
    use_mask_param_to_init_model: bool = field(default=False)
    mask_weight_scaling: float = field(default=1.0)
    topk_level: str = field(default='group_largest')
    keep_ratio: float = field(default=0.005)
    tune_bias: bool = field(default=False)

def parse_args():
    parser = HfArgumentParser(OurArgs)
    args = parser.parse_args_into_dataclasses()[0]
    print(args)
    return args

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)
    return f

def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()
    return jdict

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}


def preprocess(
    sources: Sequence[str],
    targets: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)

def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(SupervisedDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = jload(data_path)

        logging.warning("Formatting inputs...")
        prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]
        sources = [
            prompt_input.format_map(example) if example.get("input", "") != "" else prompt_no_input.format_map(example)
            for example in list_data_dict
        ]
        targets = [f"{example['output']}{tokenizer.eos_token}" for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = preprocess(sources, targets, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def obtain_binary_mask(args, keep_ratio, mask_dict, trainable_params_size, all_params_size, model=None):

    new_mask_dict = {}
    bias_mask = {} if args.topk_level == 'pafi' and 'opt' in args.model_name_or_path.lower() else None
    if args.topk_level == "group_largest":
        for name, mask in mask_dict.items():
            # mask = mask.float()
            if mask.size()[0] == 32001:
                mask = mask[:32000]
            keep_num = int(torch.prod(torch.tensor(mask.shape)).item() * keep_ratio)
            assert keep_num > 0
            top_pos = torch.topk(mask.view(-1).abs(), keep_num, largest=True)[1]
            binary_mask = torch.zeros_like(mask.view(-1))
            binary_mask[top_pos] = 1.0
            trainable_params_size += int(binary_mask.sum().item())
            name = '.'.join(name.split('.')[:-1])
            if mask.size()[0] == 32000:
                new_mask_dict[name] = torch.cat([binary_mask.reshape(mask.shape), torch.zeros(size=(1, mask.size()[1]))], dim=0)
            new_mask_dict[name] = binary_mask.reshape(mask.shape)

    elif args.topk_level == "global_largest":
        all_masks = []
        sizes = {}
        all_mask_size = 0
        for name, mask in mask_dict.items():
            # mask = mask.float()
            all_masks.append(mask.view(-1))
            name = '.'.join(name.split('.')[:-1])
            sizes[name] = mask.shape
            all_mask_size += torch.prod(torch.tensor(mask.shape)).item()
        all_masks = torch.cat(all_masks, dim=0)
        keep_num = int(all_mask_size * keep_ratio)
        assert keep_num > 0
        top_pos = torch.topk(all_masks, keep_num, largest=False)[1]
        binary_masks = torch.zeros_like(all_masks, device=all_masks.device)
        binary_masks[top_pos] = 1
        assert binary_masks.long().sum() == len(top_pos)
        trainable_params_size += int(binary_masks.sum().item())
        now_idx = 0
        for k, v in sizes.items():
            end_idx = now_idx + torch.prod(torch.tensor(v))
            new_mask_dict[k] = binary_masks[now_idx: end_idx].reshape(v).to(all_masks.device)
            now_idx = end_idx

        assert now_idx == len(binary_masks)

    elif args.topk_level == "pafi":
        embed_param_num = 0
        gate_param_num = 0
        layer_norm_param_num = 0
        other_param_num = 0
        for name, param in model.named_parameters():
            if "embed" in name or "lm_head" in name:
                embed_param_num += torch.prod(torch.tensor(param.shape)).item()
            elif "mlp.gate.weight" in name:
                gate_param_num += torch.prod(torch.tensor(param.shape)).item()
            elif "layer_norm" in name or 'layernorm' in name or '.norm.' in name:
                layer_norm_param_num += torch.prod(torch.tensor(param.shape)).item()
            else:
                other_param_num += torch.prod(torch.tensor(param.shape)).item()

        keep_num = args.keep_ratio * all_params_size
        rest_keep_num = keep_num - layer_norm_param_num
        rest_keep_ratio = rest_keep_num / other_param_num

        for name, param in model.named_parameters():

            if "embed" in name or "lm_head" in name:
                binary_mask = torch.zeros_like(param)
                name = '.'.join(name.split('.')[:-1])
                new_mask_dict[name] = binary_mask
            elif "layer_norm" in name or 'layernorm' in name or '.norm.' in name:
                binary_mask = torch.ones_like(param)
                name = '.'.join(name.split('.')[:-1])
                new_mask_dict[name] = binary_mask
                trainable_params_size += int(binary_mask.sum().item())
            elif "mlp.gate.weight" in name:
                continue
            else:
                param_size = torch.prod(torch.tensor(param.shape)).item()
                param_keep_num = int(rest_keep_ratio * param_size)
                top_pos = torch.topk(param.view(-1).abs(), param_keep_num, largest=False)[1]
                binary_mask = torch.zeros_like(param.view(-1))
                binary_mask[top_pos] = 1.0
                trainable_params_size += int(binary_mask.sum().item())
                if 'bias' in name:
                    bias_mask[name] = binary_mask.reshape(param.shape)
                else:
                    name = '.'.join(name.split('.')[:-1])
                    new_mask_dict[name] = binary_mask.reshape(param.shape)

    elif args.topk_level == "random":
        embed_param_num = 0
        gate_param_num = 0
        layer_norm_param_num = 0
        other_param_num = 0
        for name, param in model.named_parameters():
            if "embed" in name or "lm_head" in name:
                embed_param_num += torch.prod(torch.tensor(param.shape)).item()
            elif "mlp.gate.weight" in name:
                gate_param_num += torch.prod(torch.tensor(param.shape)).item()
            elif "layer_norm" in name or 'layernorm' in name or '.norm.' in name:
                layer_norm_param_num += torch.prod(torch.tensor(param.shape)).item()
            else:
                other_param_num += torch.prod(torch.tensor(param.shape)).item()

        keep_num = args.keep_ratio * all_params_size
        rest_keep_ratio = keep_num / other_param_num

        for name, param in model.named_parameters():

            if "embed" in name or "lm_head" in name:
                continue
            elif "mlp.gate.weight" in name:
                continue
            elif "layer_norm" in name or 'layernorm' in name or '.norm.' in name:
                continue
            else:
                param_size = torch.prod(torch.tensor(param.shape)).item()
                param_keep_num = int(rest_keep_ratio * param_size)
                binary_mask = torch.zeros_like(param).view(-1)
                pos = np.array(list(range(0, param_size)))
                random_pos = np.random.choice(pos, param_keep_num, replace=False)
                binary_mask[random_pos] = 1
                binary_mask = binary_mask.view(param.shape)
                name = '.'.join(name.split('.')[:-1])
                new_mask_dict[name] = binary_mask
                trainable_params_size += int(binary_mask.sum().item())

    return new_mask_dict, bias_mask, trainable_params_size

def create_optimizer(args, model, lr):
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight", ".norm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if
                       not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer, optimizer_grouped_parameters

def _setattr(model, name, module):
    name_list = name.split(".")
    for name in name_list[:-1]:
        model = getattr(model, name)
    setattr(model, name_list[-1], module)

def merge_model_weight(model, pretrained_model):
    for name, module in model.named_modules():
        if hasattr(module, 'tunable_weights'):
            for name1, module1 in pretrained_model.named_modules():
                if name == name1:
                    module1.weight.data = csr_add(
                                                 module.tunable_weights.data,
                                                 module.row_offsets.data,
                                                 module.row_indices.data,
                                                 module.col_indices.data,
                                                 module1.weight.data
                                            )
                    # replace original module by new module
                    _setattr(pretrained_model, name, module1)
    return pretrained_model


def train(args, model, optimizer, tokenizer, config, lr_scheduler, accelerator, train_dataloader, train_dataset, pretrained_model):

    total_batch_size = args.per_device_train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    # mask_step = 50

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    completed_steps = 0
    best_eval_loss = 99999999
    # update the progress_bar if load from checkpoint
    progress_bar.update(completed_steps)

    for epoch in range(0, args.num_train_epochs):
        model.train()
        for step, batch in enumerate(train_dataloader):

            if step < 1:
                for k, v in batch.items():
                    print(k, v[0])

            outputs = model(**batch)
            loss = outputs.loss
            accelerator.backward(loss)

            if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
                completed_steps += 1

                if completed_steps % 10 == 0:
                    logger.info(f"step:{completed_steps}, lr: {optimizer.param_groups[0]['lr']}, loss: {loss.item()}")
                # if args.save_steps is not None and completed_steps > 0 and completed_steps % args.save_steps == 0:
                #     accelerator.wait_for_everyone()
                #     unwrapped_model = accelerator.unwrap_model(model)
                #     if accelerator.is_main_process:
                #         unwrapped_model.to('cpu')
                #         merge_model = merge_model_weight(unwrapped_model, pretrained_model)
                #         merge_model.to(torch.bfloat16)
                #         save_mdoel_path = args.output_dir + '/epoch' + str(epoch)
                #         os.makedirs(save_mdoel_path, exist_ok=True)
                #         tokenizer.save_pretrained(save_mdoel_path)
                #         config.save_pretrained(save_mdoel_path)
                #         torch.save(merge_model.state_dict(), os.path.join(save_mdoel_path, 'pytorch_model.bin'))

            if completed_steps >= args.max_train_steps:
                break

        # eval_loss, best_eval_loss = evaluate(args, best_eval_loss, ft_model, model, tokenizer, config, eval_dataloader, accelerator)
        # logger.info(f"eval loss: {eval_loss}, best eavl loss: {best_eval_loss}")
        if epoch == args.num_train_epochs - 1:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                unwrapped_model.to('cpu')
                merge_model = merge_model_weight(unwrapped_model, pretrained_model)
                merge_model.to(torch.bfloat16)
                save_mdoel_path = args.output_dir + '/epoch' + str(epoch)
                os.makedirs(save_mdoel_path, exist_ok=True)
                tokenizer.save_pretrained(save_mdoel_path)
                config.save_pretrained(save_mdoel_path)
                torch.save(merge_model.state_dict(), os.path.join(save_mdoel_path, 'pytorch_model.bin'))



def main():
    args = parse_args()
    transformers.utils.logging.set_verbosity_info()

    accelerator = Accelerator()

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    logger.info(f"Training/evaluation parameters {args}")
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_name_or_path, trust_remote_code=True)


    if args.model_name_or_path:
        if 'llama' in args.model_name_or_path:
            tokenizer = LlamaTokenizer.from_pretrained(
                args.model_name_or_path, use_fast=True, model_max_length=args.max_training_seq_length,
                unk_token='<unk>', pad_token='<pad>', bos_token='<s>', eos_token='</s>')
        elif 'mistral' in args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, use_fast=True, model_max_length=args.max_training_seq_length,
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        else:
            tokenizer = AutoTokenizer.from_pretrained(
                args.model_name_or_path, use_fast=True, model_max_length=args.max_training_seq_length,
                trust_remote_code=True
            )
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )

    if args.model_name_or_path:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if "deepseek" in args.model_name_or_path.lower() else torch.float16,
            from_tf=bool(".ckpt" in args.model_name_or_path),
            config=config,
            trust_remote_code=True
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    #Masking
    for n, p in model.named_parameters():
        logger.info(n)

    mask_dict = {}
    if args.mask_param_path is not None:
        mask_params_dict = torch.load(args.mask_param_path, map_location='cpu')
        for name, param in mask_params_dict.items():
            # param = param.float().to('cpu')
            mask_params_dict[name] = param
            if "weight_U" in name:
                weight_V = mask_params_dict[name.replace('weight_U', 'weight_V')]
                mask_dict[name.replace('weight_U', 'weight')] = torch.mm(param, weight_V)

        logger.info(f"mask_dict: {mask_dict.keys()}")

        param_count = 0
        if args.use_mask_param_to_init_model:
            for name, param in model.named_parameters():
                if name in mask_dict:
                    if mask_dict[name].data.size()[0] != param.data.size()[0]:
                        logger.info(mask_dict[name].data.size())
                        logger.info(param.data.size())
                        param.data.copy_(param.data + mask_dict[name].data[:param.data.size()[0]] * args.mask_weight_scaling)
                    else:
                        param.data.copy_(param.data + mask_dict[name].data * args.mask_weight_scaling)
                    param_count += 1
            logger.info(f"  number of initialized parameters used learned mask= {param_count}")
            # logger.info(f"  mask_dict= {mask_dict.keys()}")

        logger.info(f"  number of parameter Masks= {len(mask_dict)}")
    elif args.topk_level != 'pafi':
        raise ValueError("Need the mask for efficient finetuning.")


    # Compute the top-k values in the mask as the trainable parameters
    bias_params_size = 0
    all_params_size = 0
    trainable_params_size = 0
    for name, param in model.named_parameters():
        size = torch.prod(torch.tensor(param.shape)).item()
        all_params_size += size
        if "bias" in name:
            bias_params_size += size
            if args.tune_bias:
                trainable_params_size += size

    bias_ratio = bias_params_size / all_params_size
    if args.tune_bias:
        rest_ratio = args.keep_ratio - bias_ratio
        logger.info(f"  ratio of bias parameters = {bias_ratio}")
        logger.info(f"  ratio of rest parameters = {rest_ratio}")
    else:
        rest_ratio = args.keep_ratio
        logger.info(f"  ratio of trainable parameters = {rest_ratio}")

    mask_dict, bias_mask, trainable_params_size = obtain_binary_mask(args, rest_ratio, mask_dict, trainable_params_size, all_params_size, model)

    logger.info(f"number of trainable parameters: {trainable_params_size}.")
    logger.info(f"number of model parameters: {all_params_size}.")
    logger.info(f"ratio of trainable parameters: {trainable_params_size / all_params_size}.")

    # ModelContext(tokenization_context, model, max_context_size=2048)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    train_dataset = SupervisedDataset(data_path=args.data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    logger.info(len(train_dataset))
    # logger.info(len(eval_dataset))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)
    pretrained_model = copy.deepcopy(model)
    if args.topk_level == 'pafi':
        Masking(model, mask_dict, bias_mask, logger, False, pafi=True)
    else:
        Masking(model, mask_dict, None, logger, False, pafi=False)
    model.to(torch.bfloat16)

    optimizer, _ = create_optimizer(args, model, lr=args.learning_rate)

    # Scheduler and math around the number of training steps.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps
        if overrode_max_train_steps
        else args.max_train_steps * accelerator.num_processes,
    )

    # Prepare everything with our `accelerator`.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    train(args, model, optimizer, tokenizer, config, lr_scheduler, accelerator, train_dataloader, train_dataset, pretrained_model)


if __name__ == "__main__":
    main()