# -*- coding: utf-8 -*-
# ActiveLearning/AL_Sampler.py
import numpy as np
import pandas as pd
from tqdm import tqdm


class ActiveSampler:
    """
    Active Learning Sampler
    - 혼합 쿼리 전략: RD + QT + UNC
    - 초기 샘플링(Initial sampling) 기능 포함
    """
    def __init__(self, rd_frac=0.3, qt_frac=0.3, num_bins=10, gamma=0.5, seed=None):
        self.rd_frac = rd_frac
        self.qt_frac = qt_frac
        self.num_bins = num_bins
        self.gamma = gamma
        self.seed = seed or 42
        self.rng = np.random.default_rng(self.seed)

    # ─────────────────────────────
    # 기본 샘플링 함수들
    # ─────────────────────────────
    def random_sampling(self, idx_pool, n_samples):
        if n_samples <= 0 or len(idx_pool) == 0:
            return np.array([], dtype=int)
        return self.rng.choice(idx_pool, size=min(n_samples, len(idx_pool)), replace=False)

    def stratified_quantile_sampling(self, low_values, idx_pool, n_samples):
        """
        Quantile binning + gamma weighting (rarest bin upweighting)
        """
        if n_samples <= 0 or len(idx_pool) == 0:
            return np.array([], dtype=int)

        quantiles = pd.qcut(low_values[idx_pool], q=self.num_bins, labels=False, duplicates="drop")
        unique_bins = np.unique(quantiles)
        bin_to_idx = {b: idx_pool[quantiles == b] for b in unique_bins}

        counts = np.array([len(bin_to_idx[b]) for b in unique_bins], dtype=float)

        # Gamma weighting
        if self.gamma != 0:
            weights = (counts ** self.gamma)
            probs = weights / weights.sum()
        else:
            probs = np.ones_like(counts) / len(counts)

        raw_alloc = probs * n_samples
        quota = np.floor(raw_alloc).astype(int)
        deficit = n_samples - quota.sum()
        if deficit > 0:
            add_bins = self.rng.choice(unique_bins, size=deficit, replace=True, p=probs)
            for b in add_bins:
                quota[b] += 1

        idx_sampled = []
        for b, q in zip(unique_bins, quota):
            bin_idxs = bin_to_idx[b]
            if len(bin_idxs) == 0 or q <= 0:
                continue
            q = min(q, len(bin_idxs))
            sampled = self.rng.choice(bin_idxs, size=q, replace=False)
            idx_sampled.extend(sampled)

        return np.array(idx_sampled)

    def uncertainty_sampling(self, uncertainties, idx_pool, n_samples):
        if n_samples <= 0 or uncertainties is None:
            return np.array([], dtype=int)
        order = np.argsort(uncertainties)[-n_samples:]
        return idx_pool[order]

    # ─────────────────────────────
    # 혼합 샘플링 (RD + QT + UNC)
    # ─────────────────────────────
    def sample_next(self, idx_unlabeled, low_values, uncertainties, samples_per_iter):
        """
        Combines RD + QT + UNC sampling strategies.
        """
        RD = int(samples_per_iter * self.rd_frac)
        QT = int(samples_per_iter * self.qt_frac)
        UNC = samples_per_iter - RD - QT

        rand_idx = self.random_sampling(idx_unlabeled, RD)
        qt_idx = self.stratified_quantile_sampling(low_values, idx_unlabeled, QT)

        # uncertainty sampling — optional
        if UNC > 0 and uncertainties is not None and len(uncertainties) == len(idx_unlabeled):
            unc_idx = self.uncertainty_sampling(uncertainties, idx_unlabeled, UNC)
        else:
            unc_idx = np.array([], dtype=int)

        all_idx = np.concatenate([rand_idx, qt_idx, unc_idx])
        if len(all_idx) == 0:
            return np.array([], dtype=int)
        return np.unique(all_idx)

    # ─────────────────────────────
    # 초기 샘플링 함수
    # ─────────────────────────────
    def initial_sampling(self, low_values, idx_pool, total_size, method="hybrid"):
        """
        Initial sampling before AL loop starts.
        method ∈ {"random", "quantile", "hybrid"}
        """
        n_samples = int(total_size)
        if n_samples <= 0:
            return np.array([], dtype=int)

        if method == "random":
            init_idx = self.random_sampling(idx_pool, n_samples)

        elif method == "quantile":
            init_idx = self.stratified_quantile_sampling(low_values, idx_pool, n_samples)

        elif method == "hybrid":
            # 절반 랜덤 + 절반 분위 기반
            half = n_samples // 2
            rand_idx = self.random_sampling(idx_pool, half)
            remain = np.setdiff1d(idx_pool, rand_idx)
            qt_idx = self.stratified_quantile_sampling(low_values, remain, n_samples - half)
            init_idx = np.unique(np.concatenate([rand_idx, qt_idx]))

        else:
            raise ValueError(f"Unsupported initial sampling method: {method}")

        return init_idx
