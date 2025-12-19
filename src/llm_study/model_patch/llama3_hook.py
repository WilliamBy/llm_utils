import os
from pathlib import Path
from typing import Optional
from typing import Callable, Unpack, Union
from types import MethodType

import torch
from packaging.version import Version
from transformers.utils import TransformersKwargs
from transformers.models.llama.modeling_llama import (
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
    apply_rotary_pos_emb,
    LlamaForCausalLM,
    LlamaModel
)
from transformers.cache_utils import Cache


_SAVE_DIR: Optional[Path] = None
_STEP = 0  # DECODING STEP COUNTER
_MODE = ""

def _print_shape_once(name, shape):
    if _STEP == 0:
        print(f"{name}: {shape}")


@torch.inference_mode()
def _patched_LlamaAttention_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
):
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(
        hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(
        hidden_shape).transpose(1, 2)

    # PATCHED
    global _STEP, _MODE
    if "q" in _MODE:
        torch.save(query_states.detach().cpu(), _SAVE_DIR /
                f"layer{self.layer_idx}_step{_STEP}_query.pt")
        _print_shape_once("query", query_states.shape)
    if "k" in _MODE:
        torch.save(key_states.detach().cpu(), _SAVE_DIR /
                f"layer{self.layer_idx}_step{_STEP}_key.pt")
        _print_shape_once("key", key_states.shape)
    if "v" in _MODE:
        torch.save(value_states.detach().cpu(), _SAVE_DIR /
                f"layer{self.layer_idx}_step{_STEP}_value.pt")
        _print_shape_once("value", value_states.shape)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states, key_states, cos, sin)

    # PATCHED: capture RoPE QK
    if "Q" in _MODE:
        torch.save(query_states.detach().cpu(), _SAVE_DIR /
                f"layer{self.layer_idx}_step{_STEP}_query_RoPE.pt")
        _print_shape_once("RoPE query", query_states.shape)
    if "K" in _MODE:
        torch.save(key_states.detach().cpu(), _SAVE_DIR /
                f"layer{self.layer_idx}_step{_STEP}_key_RoPE.pt")
        _print_shape_once("RoPE key", key_states.shape)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos,
                        "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

    attn_output, attn_weights = attention_interface(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    # PATCHED
    if "a" in _MODE:
        torch.save(
            attn_weights.detach().cpu(),
            _SAVE_DIR / f"layer{self.layer_idx}_step{_STEP}_attn_weights.pt",
        )
        _print_shape_once("attn_weights", attn_weights.shape)

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


@torch.inference_mode()
def _patched_LlamaModel_forward(
    self,
    *args,
    **kwargs,
):
    global _STEP
    _STEP += 1

    return LlamaModel.forward(self, *args, **kwargs)

MODE_MAP = {
    "Query": "q",
    "Key": "k",
    "Value": "v",
    "Attention Output": "o",
    "Attention Weight": "a",
    "Query RoPE": "Q",
    "Key RoPE": "K",
}

def enable_capture(model: LlamaForCausalLM, save_dir: str | os.PathLike, mode: str = "qkva") -> None:
    """
    Args:
        model: LlamaForCausalLM model
        save_dir: directory to save the captured tensors
        capture_attn_weight: whether to capture the attention weights
        mode: mode to capture the tensors, combine "q", "k", "v" , "a" to capture the specific tensors
    """
    global _SAVE_DIR, _STEP, _MODE

    _SAVE_DIR = Path(save_dir)
    _STEP = 0
    _MODE = mode

    # patched forward method for LlamaAttention
    for decoder_layer in model.model.layers:
        decoder_layer.self_attn.forward = MethodType(_patched_LlamaAttention_forward, decoder_layer.self_attn)

    # patched forward method for LlamaModel
    model.model.forward = MethodType(_patched_LlamaModel_forward, model.model)

    print(
        f"[llama3_hook] {mode} capture is enabled. Saving to: {_SAVE_DIR}")


def reset_step() -> None:
    global _STEP
    _STEP = 0


def get_step() -> int:
    global _STEP
    return _STEP


if __name__ == "__main__":
    model = LlamaForCausalLM.from_pretrained(
        "meta-llama/Llama-3-8B-Instruct", torch_dtype=torch.float16, device_map="cuda:0")

    # capture key and value only
    enable_capture(model, "./results", mode="kv")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3-8B-Instruct")
    inputs = tokenizer("hello, introduce yourself.", return_tensors="pt").to(model.device)
    _ = model(**inputs)

    print(f"Total steps: {get_step()}")
    reset_step()
