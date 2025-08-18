from __future__ import annotations
import random
from dataclasses import dataclass
from typing import List, Tuple

DNA_ALPHABET = 'ACGT'

@dataclass
class PolySimConfig:
    n_per_class: int = 200
    length: int = 120
    motif: str = 'TATA'
    motif_pos: int = 40
    substitution_rate: float = 0.05
    seed: int = 0


def _rand_base(exclude: str, rng: random.Random):
    choices = [b for b in DNA_ALPHABET if b!=exclude]
    return rng.choice(choices)


def generate_sequence(cfg: PolySimConfig, cls: int, rng: random.Random) -> str:
    # background GC preference: class 1 more GC
    bg_weights = [1,1,1,1]
    if cls==1:
        bg_weights = [1,2,2,1]
    seq = []
    bases = list(DNA_ALPHABET)
    for i in range(cfg.length):
        if i==cfg.motif_pos and cls==1:
            seq.extend(list(cfg.motif))
        if len(seq)>i:
            continue
        # sample background
        seq.append(rng.choices(bases, weights=bg_weights,k=1)[0])
    seq = seq[:cfg.length]
    # apply substitution noise
    noisy = []
    for b in seq:
        if rng.random()<cfg.substitution_rate:
            noisy.append(_rand_base(b, rng))
        else:
            noisy.append(b)
    return ''.join(noisy)


def generate_dataset(cfg: PolySimConfig) -> List[Tuple[int,str]]:
    rng = random.Random(cfg.seed)
    rows = []
    for c in [0,1]:
        for _ in range(cfg.n_per_class):
            rows.append((c, generate_sequence(cfg, c, rng).lower()))
    return rows
