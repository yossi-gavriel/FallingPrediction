"""Deterministic, component-scoped random number generators (no global RNG state)."""

from __future__ import annotations

import hashlib

import numpy as np


def component_key(component: str) -> tuple[int, ...]:
    digest = hashlib.sha256(component.encode("utf-8")).digest()
    return tuple(int.from_bytes(digest[i:i + 4], "little") for i in range(0, 16, 4))


def rng_for(seed: int, component: str) -> np.random.Generator:
    """Return a Generator uniquely determined by (seed, component)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(entropy=int(seed), spawn_key=component_key(component))))


def int_seed_for(seed: int, component: str) -> int:
    """Derive a 31-bit integer seed for libraries that require an int ``random_state``."""
    return int(rng_for(seed, component).integers(0, 2**31 - 1))
