"""Extract GPT-2 XL features for fMRI encoding.

For every word we build a 10-gram (the word plus the preceding words), run it
through GPT-2 XL, and take the hidden state of the final token at a chosen
layer. Per-word vectors are Lanczos-downsampled onto the fMRI TR timeline,
trimmed/z-scored, then expanded with FIR delays.

Simplified from `neuro/features/{feature_spaces,feature_utils}.py` and
`neuro/data/interp_data.py`.
"""
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# ----------------------------- ngrams -----------------------------
def get_ngrams(words: List[str], ngram_size: int = 10) -> List[str]:
    """Each word -> a string of the (ngram_size) words leading up to and incl it."""
    ngrams = []
    for i in range(len(words)):
        lo = max(0, i - ngram_size)
        ngrams.append(' '.join(words[lo: i + 1]).strip())
    return ngrams


# ----------------------------- embedder -----------------------------
class GPT2Embedder:
    """Wraps GPT-2 XL to return the final-token hidden state at a given layer."""

    def __init__(self, checkpoint='gpt2-xl', layer=24, device='cuda'):
        self.layer = layer
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModel.from_pretrained(
            checkpoint, output_hidden_states=True, torch_dtype=torch.float16
        ).to(device).eval()

    @torch.no_grad()
    def __call__(self, texts: List[str], batch_size: int = 16) -> np.ndarray:
        embs = []
        for i in tqdm(range(0, len(texts), batch_size), desc='embedding'):
            batch = texts[i: i + batch_size]
            inputs = self.tokenizer(
                batch, return_tensors='pt', padding=True, truncation=True, max_length=128
            ).to(self.device)
            # (n_layers+1) tuple of (batch, tokens, hidden)
            hidden = self.model(**inputs).hidden_states[self.layer]
            # gather the last non-pad token for each sequence (robust to padding side)
            last_idx = inputs['attention_mask'].sum(dim=1) - 1
            emb = hidden[torch.arange(hidden.shape[0]), last_idx]
            embs.append(emb.float().cpu().numpy())
        return np.concatenate(embs, axis=0)


# ----------------------------- downsampling -----------------------------
def _lanczosfun(cutoff, t, window=3):
    t = t * cutoff
    val = window * np.sin(np.pi * t) * np.sin(np.pi * t / window) / (np.pi ** 2 * t ** 2)
    val[t == 0] = 1.0
    val[np.abs(t) > window] = 0.0
    return val


def lanczos_downsample(data, oldtime, newtime, window=3):
    """Interpolate rows of `data` from `oldtime` onto `newtime` (Lanczos filter)."""
    cutoff = 1 / np.mean(np.diff(newtime))
    sincmat = np.zeros((len(newtime), len(oldtime)))
    for i in range(len(newtime)):
        sincmat[i, :] = _lanczosfun(cutoff, newtime[i] - oldtime, window)
    return np.dot(sincmat, data)


# ----------------------------- normalization / delays -----------------------------
def _zscore(v):
    s = v.std(0)
    s[s == 0] = 1.0
    return (v - v.mean(0)) / s


def trim_and_zscore(downsampled: Dict[str, np.ndarray], trim=5):
    """Trim [5+trim : -trim] per story, z-score, then stack. Matches response trim."""
    feats = [_zscore(downsampled[s][5 + trim: -trim]) for s in downsampled]
    return np.vstack(feats)


def make_delayed(stim, ndelays=4):
    """Concatenate FIR-delayed copies of `stim` (delays 1..ndelays TRs)."""
    n, d = stim.shape
    out = []
    for delay in range(1, ndelays + 1):
        dstim = np.zeros((n, d))
        dstim[delay:, :] = stim[:-delay, :]
        out.append(dstim)
    return np.hstack(out)


# ----------------------------- top-level -----------------------------
def get_features(wordseqs, stories, embedder, ngram_size=10, ndelays=4):
    """Return delayed feature matrix (sum_of_trimmed_trs, ndelays * hidden_dim)."""
    downsampled = {}
    for story in stories:
        ws = wordseqs[story]
        ngrams = get_ngrams(list(ws.data), ngram_size=ngram_size)
        word_vectors = embedder(ngrams)
        downsampled[story] = lanczos_downsample(
            word_vectors, oldtime=ws.data_times, newtime=ws.tr_times
        )
    feats = trim_and_zscore(downsampled)
    return make_delayed(feats, ndelays=ndelays)
