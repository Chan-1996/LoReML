import torch
from torch import nn
from torch.nn import functional as F
from spops import csr_add, sddmm, csr_transpose
from transformers.models.bart.modeling_bart import BartLearnedPositionalEmbedding
from transformers.models.opt.modeling_opt import OPTLearnedPositionalEmbedding
from transformers.models.mbart.modeling_mbart import MBartLearnedPositionalEmbedding
from typing import Optional
from transformers import LlamaForCausalLM


def find_module(root_module: nn.Module, key: str):
    """
    Find a module with a specific name in a Transformer model
    From OpenDelta https://github.com/thunlp/OpenDelta
    """
    sub_keys = key.split(".")
    parent_module = root_module
    for sub_key in sub_keys[:-1]:
        parent_module = getattr(parent_module, sub_key)
    module = getattr(parent_module, sub_keys[-1])
    return parent_module, sub_keys[-1], module


def get_sparse_csr(mask, out_dim, fp16, device):

    # Find indices of non-zero values
    non_zero_indices = torch.nonzero(mask, as_tuple=True)
    row_indices, col_indices = non_zero_indices

    # Extract values using the mask
    values = torch.zeros(row_indices.size(0))

    # Compute row_offsets
    row_counts = torch.bincount(row_indices, minlength=out_dim)
    row_offsets = torch.cat((torch.tensor([0]), torch.cumsum(row_counts, dim=0)))

    # col_idx is already found as col_indices
    col_idx = col_indices

    # Sort rows by the number of non-zeros
    row_idx = torch.argsort(row_counts, descending=True)

    # Conversion to the desired tensors and types
    if fp16:
        values = values.to(torch.float16).to(device)  # Ensure values are float
        row_offsets = row_offsets.to(torch.int32).to(device)  # Ensure row_offsets is int32
        col_idx = col_idx.to(torch.int16).to(device)  # Ensure col_idx is int16
        row_idx = row_idx.to(torch.int16).to(device)  # Ensure row_idx is int16, if used
    else:
        values = values.to(torch.float32).to(device)  # Ensure values are float
        row_offsets = row_offsets.to(torch.int32).to(device)  # Ensure row_offsets is int32
        col_idx = col_idx.to(torch.int16).to(device)  # Ensure col_idx is int16
        row_idx = row_idx.to(torch.int16).to(device)  # Ensure row_idx is int16, if used

    # Creating a dummy sparse_weight object to mimic returning a similar object as the original function
    sparse_weight = {
        'values': values,
        'row_offsets': row_offsets,
        'col_indices': col_idx,
        'row_indices': row_idx,
    }

    return sparse_weight


class SparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, row_indices, col_indices, dense_weight, bias, input, bias_mask=None):
        # Save tensors for backward pass
        ctx.save_for_backward(values, row_offsets, row_indices, col_indices, dense_weight, bias, input, bias_mask)

        # Perform the sparse addition
        total_weight = csr_add(values, row_offsets, row_indices, col_indices, dense_weight)

        # Perform the linear operation
        output = F.linear(input, total_weight, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        values, row_offsets, row_indices, col_indices, dense_weight, bias, input, bias_mask = ctx.saved_tensors
        input_shape = input.shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]) # (batch * seq, hidden_size)
        input = input.reshape(-1, input.shape[-1]) # (batch * seq, hidden_size)

        grad_values = sddmm(row_offsets, row_indices, col_indices, grad_output.T.contiguous(), input.T.contiguous())

        grad_input = (grad_output @ csr_add(values, row_offsets, row_indices, col_indices, dense_weight)).reshape(
            input_shape)
        if bias is not None:
            grad_bias = grad_output.sum(0)
            if bias_mask is not None:
                grad_bias *= bias_mask.to(grad_bias.device)
        else:
            grad_bias = None
        # grad_bias = None

        return grad_values, None, None, None, None, grad_bias, grad_input, None


class SparseEmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, values, row_offsets, row_indices, col_indices, dense_weight, input, padding_idx):
        # Save tensors for backward pass
        ctx.save_for_backward(values, row_offsets, row_indices, col_indices, dense_weight, input)

        total_weight = csr_add(values, row_offsets, row_indices, col_indices, dense_weight)

        # Perform the linear operation
        output = total_weight[input]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        values, row_offsets, row_indices, col_indices, dense_weight, input = ctx.saved_tensors
        # input_shape = input.shape
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]) # (batch * seq, hidden_size)
        input = input.reshape(-1, input.shape[-1]) # (batch * seq, hidden_size)

        grad_values = sddmm(row_offsets, row_indices, col_indices, grad_output.T.contiguous(), input.T.contiguous())

        # grad_input = (grad_output @ csr_add(values, row_offsets, row_indices, col_indices, dense_weight)).reshape(
        #     input_shape)

        return grad_values, None, None, None, None, None, None, None

class EmbeddingFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, dense_weight, input, mask):
        # Save tensors for backward pass
        ctx.save_for_backward(dense_weight, input, mask)
        # Perform the embedding lookup
        output = dense_weight[input]
        return output

    @staticmethod
    def backward(ctx, grad_output):
        dense_weight, input, mask = ctx.saved_tensors
        input_one_hot = F.one_hot(input, num_classes=dense_weight.shape[0]).to(
            input.device).float()  # (batch, seq, vocab_size)
        grad_output = grad_output.reshape(-1, grad_output.shape[-1]) # (batch * seq, hidden_size)
        input_one_hot = input_one_hot.reshape(-1, input_one_hot.shape[-1]).to_sparse() # (batch * seq, vocab_size)

        grad_values = torch.sparse.mm(input_one_hot.transpose(),  grad_output.transpose())
        masked_grad_values = grad_values * mask

        return masked_grad_values, None, None


class MaskingLinear(torch.nn.Module):
    def __init__(self, base_Linear: nn.Linear, in_dim: int, out_dim: int, mask: dict, bias_mask: dict, fp16: bool):
        super().__init__()
        self.base_Linear = base_Linear
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bias_mask = bias_mask
        self.fp16 = fp16

        # Initialization of the sparse matrix components
        sparse_csr = get_sparse_csr(mask, out_dim, fp16, base_Linear.weight.device)
        self.tunable_weights = nn.Parameter(sparse_csr['values'].to(base_Linear.weight.device))

        # Register row_offsets and col_indices as buffers
        self.register_buffer('row_offsets', sparse_csr['row_offsets'])
        self.register_buffer('col_indices', sparse_csr['col_indices'])
        self.register_buffer('row_indices', sparse_csr['row_indices'])

    def forward(self, input):
        if hasattr(self.base_Linear, 'bias'):
            return SparseLinearFunction.apply(self.tunable_weights, self.row_offsets, self.row_indices, self.col_indices,
                                          self.base_Linear.weight, self.base_Linear.bias, input, self.bias_mask)
        else:
            return SparseLinearFunction.apply(self.tunable_weights, self.row_offsets, self.row_indices,
                                              self.col_indices,
                                              self.base_Linear.weight, None, input, self.bias_mask)

class MaskingEmbedding(torch.nn.Module):
    def __init__(self, base_Embedding: nn.Embedding, out_dim: int, mask: torch.tensor, fp16: bool, padding_idx: Optional[int] = None):
        super().__init__()
        self.base_Embedding = base_Embedding
        self.out_dim = out_dim
        self.fp16 = fp16
        self.padding_idx = padding_idx
        self.mask = mask

    def forward(self, input):
        # # input_one_hot = F.one_hot(input, num_classes=self.base_Embedding.weight.shape[0]).to(input.device).float()  # (batch, seq, vocab_size)
        # return SparseEmbeddingFunction.apply(self.tunable_weights, self.row_offsets, self.row_indices, self.col_indices,
        #                                   self.base_Embedding.weight, input, self.padding_idx)
        return EmbeddingFunction.apply(self.base_Embedding.weight, input, self.mask.bool())


class Masking:
    def __init__(self, model, mask_dict, bias_mask_dict, logger, fp16, pafi=False):
        self.model = model
        self.mask_dict = mask_dict
        self.bias_mask_dict = bias_mask_dict
        self.fp16 = fp16
        self.pafi = pafi
        self.logger = logger

        module_list = self.mask_dict.keys()
        self.logger.info(module_list)

        for key, _ in self.model.named_modules():

            if key in module_list and 'embed' not in key and "lm_head" not in key:
                if self.pafi:
                    if 'layer_norm' in key or 'layernorm' in key or 'norm' in key:
                        continue
                    if self.bias_mask_dict is not None:
                        bias_mask = self.bias_mask_dict[key + '.bias']
                    else:
                        bias_mask = None
                else:
                    bias_mask = None
                self.logger.info(key, main_process_only=True)
                module_name = key.split(".")[-1]
                parent_module, sub_key, module = find_module(self.model, key)
                if self.model.config.model_type == 'roberta':
                    if 'intermediate.dense' in key:
                        out_dim = self.model.config.intermediate_size
                    else:
                        out_dim = self.model.config.hidden_size
                    if 'output.dense' in key and 'attention' not in key:
                        in_dim = self.model.config.intermediate_size
                    else:
                        in_dim = self.model.config.hidden_size
                else:
                    if module_name == 'fc1':
                        out_dim = self.model.config.ffn_dim
                    elif module_name == 'gate_proj' or module_name == 'up_proj':
                        out_dim = self.model.config.intermediate_size
                    else:
                        out_dim = self.model.config.hidden_size
                    if module_name == 'fc2':
                        in_dim = self.model.config.ffn_dim
                    elif module_name == 'down_proj':
                        in_dim = self.model.config.intermediate_size
                    else:
                        in_dim = self.model.config.hidden_size
                    if self.model.config.model_type in ["llama", "deepseek", "qwen2", "qwen3"] and module_name == 'lm_head':
                        out_dim = self.model.config.vocab_size

                setattr(parent_module, sub_key, MaskingLinear(base_Linear=module, in_dim=in_dim, out_dim=out_dim,
                                                              mask=mask_dict[key], bias_mask=bias_mask, fp16=self.fp16))
            elif key in module_list and ('embed' in key or "lm_head" in key):
                if self.pafi:
                    continue
                self.logger.info(key, main_process_only=True)
                parent_module, sub_key, module = find_module(self.model, key)
                if isinstance(module, BartLearnedPositionalEmbedding) or isinstance(module, OPTLearnedPositionalEmbedding)\
                        or isinstance(module, MBartLearnedPositionalEmbedding):
                    continue
                if "lm_head" in key and self.model.config.model_type in ["qwen2", "qwen3"]:
                   continue

                out_dim = self.model.config.hidden_size
                # print(module)
                if 'token_type' in key:
                    setattr(parent_module, sub_key, MaskingEmbedding(base_Embedding=module, out_dim=out_dim,
                                                                  mask=mask_dict[key], fp16=self.fp16, padding_idx=None))
                else:
                    setattr(parent_module, sub_key, MaskingEmbedding(base_Embedding=module, out_dim=out_dim,
                                                                         mask=mask_dict[key], fp16=self.fp16,
                                                                         padding_idx=self.model.config.pad_token_id))

        for n, p in model.named_parameters():
            if "tunable_weights" not in n:
                if 'bias' in n:
                    p.requires_grad = True
                else:
                    if self.pafi and ('layer_norm' in n or 'layernorm' in n or 'norm' in n):
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
            else:
                p.requires_grad = True
