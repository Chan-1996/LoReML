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
import gc
import json
import io
from typing import Optional, Dict, Sequence
from lora.utils import lora_state_dict, convert_lora_network, convert_lora_network_att_qv
import datasets
#
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

    save_steps: int = field(default=4000)

    #lora
    weight_rank: int = field(default=16)
    alpha: int = field(default=16)



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


def create_optimizer(args, model, lr):
    no_decay = ["bias", "layer_norm.weight", "layernorm.weight", ".norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if 'weight_U' in n or 'weight_V' in n],
            "weight_decay": args.weight_decay
        },
    ]
    for name, param in model.named_parameters():
        if 'weight_U' not in name and 'weight_V' not in name:
            param.requires_grad = False

    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)
    return optimizer, optimizer_grouped_parameters

def train(args, model, optimizer, tokenizer, config, lr_scheduler, accelerator, train_dataloader, train_dataset):

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
                #         save_mdoel_path = args.output_dir + '/steps' + str(completed_steps)
                #         os.makedirs(save_mdoel_path, exist_ok=True)
                #         tokenizer.save_pretrained(save_mdoel_path)
                #         config.save_pretrained(save_mdoel_path)
                #         torch.save(unwrapped_model.state_dict(), os.path.join(save_mdoel_path, 'pytorch_model.bin'))

            if completed_steps >= args.max_train_steps:
                break

        # eval_loss, best_eval_loss = evaluate(args, best_eval_loss, ft_model, model, tokenizer, config, eval_dataloader, accelerator)
        # logger.info(f"eval loss: {eval_loss}, best eavl loss: {best_eval_loss}")
        if args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            if accelerator.is_main_process:
                save_mdoel_path = args.output_dir + '/epoch' + str(epoch)
                os.makedirs(save_mdoel_path, exist_ok=True)
                tokenizer.save_pretrained(save_mdoel_path)
                config.save_pretrained(save_mdoel_path)
                torch.save(unwrapped_model.state_dict(), os.path.join(save_mdoel_path, 'pytorch_model.bin'))
                torch.save(lora_state_dict(unwrapped_model), os.path.join(args.output_dir,
                                                                          'lora.bin'))



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
                args.model_name_or_path, use_fast=True, model_max_length=args.max_training_seq_length, trust_remote_code=True
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
            trust_remote_code=True,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForCausalLM.from_config(config)

    # ModelContext(tokenization_context, model, max_context_size=2048)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))
    # if model.config.decoder_start_token_id is None:
    #     raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    # Convert original model to lora model
    logger.info("***** Convert lora network *****")
    if 'opt' in args.model_name_or_path:
        lora_module = model.model
    else:
        lora_module = model

    convert_lora_network_att_qv(lora_module, args.weight_rank, args.alpha, logger=logger)

    all_params_size = 0
    introduce_params_size = 0
    for name, param in model.named_parameters():
        size = torch.prod(torch.tensor(param.shape)).item()
        if "weight_U" in name or "weight_V" in name:
            introduce_params_size += size
        all_params_size += size

    logger.info(f"number of lora parameters: {introduce_params_size}.")
    logger.info(f"number of model parameters: {all_params_size}.")
    logger.info(f"ratio of lora parameters: {introduce_params_size / all_params_size}.")

    train_dataset = SupervisedDataset(data_path=args.data_path, tokenizer=tokenizer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    logger.info(len(train_dataset))
    # logger.info(len(eval_dataset))

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, collate_fn=data_collator, batch_size=args.per_device_train_batch_size
    )
    # eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=args.per_device_eval_batch_size)

    # model.model.gradient_checkpointing = True
    # model.gradient_checkpointing_enable()
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

    train(args, model, optimizer, tokenizer, config, lr_scheduler, accelerator, train_dataloader, train_dataset)


if __name__ == "__main__":
    main()