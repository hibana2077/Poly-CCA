import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

DNA_ALPHABET = "ACGT"

_label_clean_re = re.compile(r"[^A-Za-z0-9_]+")


def _clean_token(tok: str) -> str:
    return _label_clean_re.sub("", tok)


def read_promoter_dataset(root: Path) -> List[Tuple[int, str]]:
    """Read UCI promoter dataset.

    Returns list of (label, sequence) where label: 1=promoter, 0=non.
    """
    data_file = root / "molecular+biology+promoter+gene+sequences" / "promoters.data"
    rows: List[Tuple[int,str]] = []
    for line in data_file.read_text().splitlines():
        line=line.strip()
        if not line:
            continue
        parts = [p for p in line.split(',') if p.strip()]
        if len(parts) < 2:
            continue
        label_raw = parts[0].strip()
        seq = parts[-1].strip().lower().replace('u','t')
        if set(seq)-set('acgt'):
            # remove non ACGT chars
            seq = ''.join([c for c in seq if c in 'acgt'])
        label = 1 if label_raw.startswith('+') else 0
        rows.append((label, seq))
    return rows


def read_splice_dataset(root: Path) -> List[Tuple[int, str]]:
    """Read UCI splice-junction dataset.

    Labels: EI, IE, N (or anything else) mapped to ints 0..C-1.
    We'll map: EI->0, IE->1, anything else->2.
    """
    data_file = root / "molecular+biology+splice+junction+gene+sequences" / "splice.data"
    rows: List[Tuple[int,str]] = []
    for line in data_file.read_text().splitlines():
        line=line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(',') if p.strip()]
        if len(parts) < 2:
            continue
        label_token = parts[0]
        seq = parts[-1].replace(' ','').upper().replace('U','T')
        # remove any non-ACGT
        seq = ''.join([c for c in seq if c in DNA_ALPHABET])
        if label_token == 'EI':
            y=0
        elif label_token == 'IE':
            y=1
        else:
            y=2
        rows.append((y, seq.lower()))
    return rows


def make_kmer_index(k: int):
    from itertools import product
    kmers = [''.join(p) for p in product(DNA_ALPHABET, repeat=k)]
    return {kmer:i for i,kmer in enumerate(kmers)}


def kmer_count(seq: str, k: int, index=None):
    if index is None:
        index = make_kmer_index(k)
    import numpy as np
    counts = np.zeros(len(index), dtype=float)
    for i in range(len(seq)-k+1):
        kmer = seq[i:i+k]
        idx = index.get(kmer)
        if idx is not None:
            counts[idx]+=1
    if counts.sum()>0:
        counts /= counts.sum()
    return counts


def batch_kmer_counts(seqs: List[str], k: int):
    index = make_kmer_index(k)
    import numpy as np
    X = np.vstack([kmer_count(s,k,index) for s in seqs])
    return X, index


def polynomial_features(X, degree: int = 2, include_bias: bool=False):
    from itertools import combinations_with_replacement
    import numpy as np
    n_samples, n_features = X.shape
    cols = []
    # degree 1
    if degree>=1:
        cols.append(X)
    if degree>=2:
        for i,j in combinations_with_replacement(range(n_features),2):
            prod = (X[:,i]*X[:,j])[:,None]
            cols.append(prod)
    Phi = np.hstack(cols)
    if include_bias:
        Phi = np.hstack([np.ones((n_samples,1)), Phi])
    return Phi


def perturb_substitution(seqs: List[str], p: float, rng=None) -> List[str]:
    import random
    if rng is None:
        rng = random.Random(0)
    out = []
    for s in seqs:
        chars = list(s)
        for i,c in enumerate(chars):
            if rng.random() < p:
                choices = [b for b in DNA_ALPHABET.lower() if b!=c]
                chars[i]=rng.choice(choices)
        out.append(''.join(chars))
    return out


def perturb_indel(seqs: List[str], p_ins: float = 0.01, p_del: float = 0.01, rng=None, max_insert: int = 1) -> List[str]:
    """Apply simple independent insertion/deletion noise.

    For each position, with probability p_del delete the base; with probability p_ins insert up to max_insert random bases after it.
    Keeps sequence roughly similar length for small probabilities.
    """
    import random
    if rng is None:
        rng = random.Random(0)
    out = []
    bases = list(DNA_ALPHABET.lower())
    for s in seqs:
        new_chars = []
        for ch in s:
            # deletion
            if rng.random() < p_del:
                continue
            new_chars.append(ch)
            # insertion after
            if rng.random() < p_ins:
                ins_len = 1 if max_insert == 1 else rng.randint(1, max_insert)
                for _ in range(ins_len):
                    new_chars.append(rng.choice(bases))
        out.append(''.join(new_chars))
    return out
