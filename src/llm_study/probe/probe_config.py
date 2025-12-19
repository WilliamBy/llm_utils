from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, Iterator, List, Literal, Optional, Tuple


TensorKind = Literal["q", "k", "v", "attn", "hidden", "logits", "custom"]


@dataclass
class HeadLocation:
    """
    Location of a specific (group, head) inside a transformer block.

    Indices are zero-based to be consistent with PyTorch and most HF models.
    """

    layer: int
    """Index of the transformer layer (block)."""

    group: Optional[int] = None
    """Optional index of the attention head *group* (for GQA / MQA style models)."""

    head: Optional[int] = None
    """Optional index of the head inside the group (or inside all heads if group is None)."""

    def __str__(self) -> str:
        return f"HeadLocation(layer={self.layer}, group={self.group}, head={self.head})"


@dataclass
class ProbeItem:
    """
    Single probe description for one tensor we want to watch.

    This struct is intentionally lightweight and model-agnostic. Concrete
    hook-registration code can map (layer, group, head, kind) to actual
    module objects and hook functions.
    """

    location: HeadLocation
    """Where in the model this tensor lives (layer / group / head)."""

    kind: TensorKind
    """What tensor to capture: q / k / v / attn / hidden / logits / custom."""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a plain dict for JSON / YAML serialization."""
        data = asdict(self)
        # asdict already turns nested dataclasses into dicts
        return data

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ProbeItem":
        """Create a ProbeItem from a dict produced by `to_dict`."""
        loc_dict = data.get("location") or {}
        location = HeadLocation(**loc_dict)

        return ProbeItem(
            location=location,
            kind=data["kind"],
            enabled=data.get("enabled", True),
            meta=data.get("meta", {}) or {},
        )

class ProbeConfig:
    """
    Central configuration object for all probes used during LLM execution.

    This class does *not* depend on a specific model implementation. Instead,
    it records abstract probe specifications (layer / group / head / kind)
    that can later be consumed by code that registers PyTorch hooks on a
    concrete model instance.

    Usage example
    -------------
    >>> cfg = LLMProbeConfig()
    >>> cfg.add_probe(
    ...     name="layer3_group0_head2_q",
    ...     layer=3,
    ...     group=0,
    ...     head=2,
    ...     kind="q",
    ... )
    >>> list(cfg.iter_probes())
    [ProbeItem(...)]
    """

    def __init__(self, items: Optional[Iterable[ProbeItem]] = None) -> None:
        self._items: Dict[str, ProbeItem] = {}
        if items is not None:
            for item in items:
                self.add_item(item)

    # ------------------------------------------------------------------
    # Creation API
    # ------------------------------------------------------------------
    def add_probe(
        self,
        name: str,
        kind: TensorKind,
        layer: int,
        group: Optional[int] = None,
        head: Optional[int] = None,
    ) -> ProbeItem:
        """
        Define a new probe and add it to this config.

        Args:
            name: Unique name used as key when storing captured tensors.
            layer: Transformer layer index (0-based).
            group: Optional attention-head group index (for GQA / MQA).
            head: Optional head index.
            kind: Which tensor to capture, e.g. "q", "k", "v", "attn", ...
            enabled: Whether this probe is active by default.
            meta: Optional arbitrary metadata dict.
        """
        item = ProbeItem(
            name=name,
            location=HeadLocation(layer=layer, group=group, head=head),
            kind=kind,
        )
        self.add_item(item)
        return item

    def add_item(self, item: ProbeItem) -> None:
        """Add an already-constructed ProbeItem to this config."""
        if item.name in self._items:
            raise ValueError(f"Probe with name '{item.name}' already exists.")
        self._items[item.name] = item

    def remove_probe(self, name: str) -> None:
        """Remove a probe by name; silently ignore if it does not exist."""
        self._items.pop(name, None)

    # ------------------------------------------------------------------
    # Query API
    # ------------------------------------------------------------------
    def get(self, name: str) -> Optional[ProbeItem]:
        """Get a probe by name; returns None if not found."""
        return self._items.get(name)

    def probes_for_layer(
        self,
        layer: int,
    ) -> List[ProbeItem]:
        """Return all probes targeting a specific layer."""
        return [
            p
            for p in self._items.values()
            if p.location.layer == layer
        ]

    def probes_for_location(
        self,
        layer: int,
        group: Optional[int] = None,
        head: Optional[int] = None,
        enabled_only: bool = True,
    ) -> List[ProbeItem]:
        """
        Return probes that match a specific (layer, group, head) triple.

        If group or head is None, matching is done only on the specified
        fields (i.e. None acts as a wildcard).
        """

        def _match(p: ProbeItem) -> bool:
            loc = p.location
            if loc.layer != layer:
                return False
            if group is not None and loc.group != group:
                return False
            if head is not None and loc.head != head:
                return False
            return True

        return [p for p in self.iter_probes() if _match(p)]

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """
        Export configuration to a dict that is easy to dump as JSON / YAML.

        Example structure:
            {
                "probes": [
                    { ... ProbeItem dict ... },
                    ...
                ]
            }
        """
        return {"probes": [p.to_dict() for p in self.iter_probes()]}

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> "ProbeConfig":
        """Create a config from a dict produced by `to_dict`."""
        probes_data = data.get("probes", []) or []
        items = [ProbeItem.from_dict(d) for d in probes_data]
        return ProbeConfig(items=items)

    # ------------------------------------------------------------------
    # Convenience / dunder methods
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._items)

    def __iter__(self) -> Iterator[ProbeItem]:  # pragma: no cover - trivial
        return self.iter_probes()

    def __contains__(self, name: str) -> bool:  # pragma: no cover - trivial
        return name in self._items