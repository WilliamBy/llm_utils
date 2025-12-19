from __future__ import annotations

from typing import Dict, List, Optional

import torch
from transformers.models.llama import modeling_llama as llama_mod

from .probe_config import ProbeConfig, ProbeItem


_ORIG_EAGER_ATTENTION_FORWARD = llama_mod.eager_attention_forward
_ACTIVE_PROBER: Optional["ProberForCausalLM"] = None


class ProberForCausalLM:
    """
    Execute probes defined in `ProbeConfig` on top of an LLM model.

      1. Looks up all probes targeting the current layer.
      2. Slices the query / key / value / attn tensors according to probe
         head indices.
      3. Stores the resulting tensors in `self.records`.

    Limitations
    ----------
    - Prober assumes:
        query: [batch, num_attention_heads, q_len, head_dim]
        key  : [batch, num_kv_heads, kv_len, head_dim]
        value: [batch, num_kv_heads, kv_len, head_dim]
    - `group` and `head` semantics follow `ProbeItem.location`:
        * For kind "q" / "attn": only `head` is used (over num_q_heads).
        * For kind "k" / "v"  : only `group` is used (over num_kv_heads).
    """

    def __init__(self, config: ProbeConfig) -> None:
        self.config = config
        # Collected tensors: name -> list[Tensor], one entry per call/step.
        self.records: Dict[str, List[torch.Tensor]] = {}
        self._active: bool = False
        self._step: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def start(self) -> None:
        """
        Enable probing by monkeypatching Llama eager attention.

        This method is idempotent; calling it multiple times without
        `stop()` is a no-op. Only one active `LLMProber` is allowed at
        a time, because the monkeypatch is global.
        """
        global _ACTIVE_PROBER

        if self._active:
            return

        if _ACTIVE_PROBER is not None and _ACTIVE_PROBER is not self:
            raise RuntimeError(
                "Another LLMProber instance is already active. "
                "Please call `stop()` on it before starting a new one."
            )

        self._patch_eager_attention()
        self._active = True
        _ACTIVE_PROBER = self
        self._step = 0
        self.records.clear()

    def stop(self) -> None:
        """
        Disable probing and restore original eager attention implementation.
        """
        global _ACTIVE_PROBER

        if not self._active:
            return

        self._restore_eager_attention()
        self._active = False
        if _ACTIVE_PROBER is self:
            _ACTIVE_PROBER = None

    def reset(self) -> None:
        """
        Reset internal step counter and clear all collected records.
        """
        self._step = 0
        self.records.clear()

    # Context manager support
    def __enter__(self) -> "ProberForCausalLM":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _patch_eager_attention(self) -> None:
        """
        Monkeypatch `llama_mod.eager_attention_forward` with our wrapper.
        """

        def _wrapper(
            module: torch.nn.Module,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attention_mask: Optional[torch.Tensor],
            scaling: float,
            dropout: float = 0.0,
            **kwargs,
        ):
            """
            Wrap the original `eager_attention_forward` to capture tensors.
            """
            layer_idx = getattr(module, "layer_idx", -1)
            step = self._step
            self._step += 1

            # Capture tensors before running attention
            self._capture_qkv(layer_idx, step, query, key, value)

            # Run original implementation
            attn_output, attn_weights = _ORIG_EAGER_ATTENTION_FORWARD(
                module,
                query,
                key,
                value,
                attention_mask,
                scaling,
                dropout=dropout,
                **kwargs,
            )

            # Capture attention weights after computation
            self._capture_attn(layer_idx, step, attn_weights)

            return attn_output, attn_weights

        llama_mod.eager_attention_forward = _wrapper

    def _restore_eager_attention(self) -> None:
        """
        Restore the original `eager_attention_forward` implementation.
        """
        llama_mod.eager_attention_forward = _ORIG_EAGER_ATTENTION_FORWARD

    # ------------------------------------------------------------------
    # Capture logic
    # ------------------------------------------------------------------
    def _capture_qkv(
        self,
        layer_idx: int,
        step: int,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """
        Capture Q/K/V tensors for all probes on the given layer.
        """
        probes = self.config.probes_for_layer(layer_idx)
        if not probes:
            return

        for probe in probes:
            if probe.kind == "q":
                self._store_probe_tensor(
                    probe,
                    self._slice_query(probe, query),
                    step=step,
                )
            elif probe.kind == "k":
                self._store_probe_tensor(
                    probe,
                    self._slice_kv(probe, key),
                    step=step,
                )
            elif probe.kind == "v":
                self._store_probe_tensor(
                    probe,
                    self._slice_kv(probe, value),
                    step=step,
                )

    def _capture_attn(
        self,
        layer_idx: int,
        step: int,
        attn_weights: torch.Tensor,
    ) -> None:
        """
        Capture attention-weight tensors for all probes on the given layer.
        """
        probes = self.config.probes_for_layer(layer_idx)
        if not probes:
            return

        for probe in probes:
            if probe.kind == "attn":
                self._store_probe_tensor(
                    probe,
                    self._slice_attn(probe, attn_weights),
                    step=step,
                )

    # ------------------------------------------------------------------
    # Tensor slicing helpers
    # ------------------------------------------------------------------
    def _slice_query(self, probe: ProbeItem, query: torch.Tensor) -> torch.Tensor:
        """
        Slice query tensor according to probe location.

        Expected shape: [batch, num_q_heads, q_len, head_dim]
        - If `probe.location.head` is not None, take that head index.
        - Otherwise, return the full query tensor.
        """
        head_idx = probe.location.head
        if head_idx is None:
            return query.detach().cpu()

        return query[:, head_idx : head_idx + 1].detach().cpu()

    def _slice_kv(self, probe: ProbeItem, kv: torch.Tensor) -> torch.Tensor:
        """
        Slice key/value tensor according to probe location.

        Expected shape: [batch, num_kv_heads, kv_len, head_dim]
        - If `probe.location.group` is not None, treat it as kv-head index.
        - Otherwise, return the full tensor.
        """
        group_idx = probe.location.group
        if group_idx is None:
            return kv.detach().cpu()

        return kv[:, group_idx : group_idx + 1].detach().cpu()

    def _slice_attn(self, probe: ProbeItem, attn: torch.Tensor) -> torch.Tensor:
        """
        Slice attention-weight tensor according to probe location.

        Expected shape: [batch, num_q_heads, q_len, kv_len]
        - If `probe.location.head` is not None, slice on the head dimension.
        - Otherwise, return the full tensor.
        """
        head_idx = probe.location.head
        if head_idx is None:
            return attn.detach().cpu()

        return attn[:, head_idx : head_idx + 1].detach().cpu()

    # ------------------------------------------------------------------
    # Storage helper
    # ------------------------------------------------------------------
    def _store_probe_tensor(
        self,
        probe: ProbeItem,
        tensor: torch.Tensor,
        step: int,
    ) -> None:
        """
        Store a captured tensor in `self.records`.

        The tensor is detached and moved to CPU by the slicing helpers.
        """
        name = probe.name
        if name not in self.records:
            self.records[name] = []
        self.records[name].append(tensor)


__all__ = ["ProberForCausalLM"]

