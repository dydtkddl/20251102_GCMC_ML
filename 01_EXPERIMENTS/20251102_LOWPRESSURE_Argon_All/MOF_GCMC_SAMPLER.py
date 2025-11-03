# -*- coding: utf-8 -*-
"""
MOF_GCMC_SAMPLER.py
────────────────────────────────────────────
Reusable GCMCSampler with log-scaling, quantile/random sampling,
and integrated histogram visualization.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class GCMCSampler:
    sampler_type: str = "qt_then_rd"
    train_ratio: float = 0.8
    qt_frac: float = 0.4
    n_bins: int = 10
    gamma: float = 0.5
    seed_base: int = 42
    qt_col: Optional[str] = None
    use_log: bool = True
    log_eps: float = 1e-12
    outdir: Optional[str] = None

    # ───────────────────────────────────────────────
    def _split_random(self, n_total: int, seed: int) -> Dict[str, np.ndarray]:
        """단순 랜덤 샘플링"""
        idx = np.arange(n_total)
        rng = np.random.default_rng(seed)
        n_train = int(round(self.train_ratio * n_total))
        train_idx = rng.choice(idx, size=n_train, replace=False)
        test_idx = np.setdiff1d(idx, train_idx)
        return {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_qt_idx": None,
            "train_rd_idx": train_idx,
        }

    # ───────────────────────────────────────────────
    def _sample_quantile_weighted(self, vals: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
        """log 스케일 기반 분위 가중 샘플링"""
        rng = np.random.default_rng(seed)

        if self.use_log:
            vals = np.log10(np.clip(vals, a_min=self.log_eps, a_max=None))

        vmin, vmax = float(vals.min()), float(vals.max())
        idx = np.arange(len(vals))
        if vmin == vmax:
            return rng.choice(idx, size=min(n_samples, len(idx)), replace=False)

        edges = np.linspace(vmin, vmax, self.n_bins + 1)
        bin_ids = np.digitize(vals, edges) - 1
        bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)

        bin_to_idx = {b: idx[bin_ids == b] for b in range(self.n_bins)}
        counts = np.array([len(bin_to_idx[b]) for b in range(self.n_bins)], dtype=float)

        valid = np.where(counts > 0)[0]
        weights = np.zeros_like(counts)
        weights[valid] = counts[valid] ** self.gamma
        probs = weights / weights.sum()

        raw = probs * n_samples
        quota = np.floor(raw).astype(int)
        deficit = n_samples - quota.sum()
        if deficit > 0:
            frac = raw - quota
            add_bins = rng.choice(np.arange(self.n_bins), size=deficit, replace=True, p=frac / frac.sum())
            for b in add_bins:
                quota[b] += 1

        selected = []
        for b, q in enumerate(quota):
            if q <= 0 or len(bin_to_idx[b]) == 0:
                continue
            q = min(q, len(bin_to_idx[b]))
            selected.append(rng.choice(bin_to_idx[b], size=q, replace=False))

        if not selected:
            return np.array([], dtype=int)

        sel = np.concatenate(selected)
        if len(sel) > n_samples:
            sel = sel[:n_samples]
        return sel
    # ───────────────────────────────────────────────
    def _sample_quantile_weighted(self, vals: np.ndarray, n_samples: int, seed: int) -> np.ndarray:
        """log 스케일 기반 분위 가중 샘플링"""
        rng = np.random.default_rng(seed)

        if self.use_log:
            vals = np.log10(np.clip(vals, a_min=self.log_eps, a_max=None))

        vmin, vmax = float(vals.min()), float(vals.max())
        idx = np.arange(len(vals))
        if vmin == vmax:
            return rng.choice(idx, size=min(n_samples, len(idx)), replace=False)

        edges = np.linspace(vmin, vmax, self.n_bins + 1)
        bin_ids = np.digitize(vals, edges) - 1
        bin_ids = np.clip(bin_ids, 0, self.n_bins - 1)

        bin_to_idx = {b: idx[bin_ids == b] for b in range(self.n_bins)}
        counts = np.array([len(bin_to_idx[b]) for b in range(self.n_bins)], dtype=float)
        valid = np.where(counts > 0)[0]

        weights = np.zeros_like(counts)
        weights[valid] = counts[valid] ** self.gamma
        probs = weights / weights.sum()

        # 각 bin별 할당 개수 계산
        raw = probs * n_samples
        quota = np.floor(raw).astype(int)
        deficit = n_samples - quota.sum()

        # 부족한 개수 보정
        if deficit > 0:
            frac = raw - quota
            add_bins = rng.choice(np.arange(self.n_bins), size=deficit, replace=True, p=frac / frac.sum())
            for b in add_bins:
                quota[b] += 1

        selected = []
        for b, q in enumerate(quota):
            if q <= 0 or len(bin_to_idx[b]) == 0:
                continue
            q = min(q, len(bin_to_idx[b]))
            selected.append(rng.choice(bin_to_idx[b], size=q, replace=False))

        # 최종 개수 보정 (정확히 n_samples 맞추기)
        if not selected:
            return np.array([], dtype=int)
        sel = np.concatenate(selected)
        if len(sel) > n_samples:
            sel = sel[:n_samples]
        elif len(sel) < n_samples:
            # 부족한 만큼 랜덤으로 추가 (아직 선택 안된 idx 중에서)
            remain = np.setdiff1d(idx, sel)
            add_n = n_samples - len(sel)
            add = rng.choice(remain, size=min(add_n, len(remain)), replace=False)
            sel = np.concatenate([sel, add])

        return np.unique(sel)
    def _split_qt_then_rd(self, df: pd.DataFrame, seed_qt: int, seed_rd: int) -> Dict[str, np.ndarray]:
        """분위 + 랜덤 혼합 샘플링 (qt_frac: 전체 중 분위 샘플 비율 + 히스토그램 저장)"""
        if self.qt_col is None or self.qt_col not in df.columns:
            raise KeyError(f"qt_then_rd requires '{self.qt_col}' column to exist.")

        n_total = len(df)
        idx_all = np.arange(n_total)
        vals = df[self.qt_col].astype(float).values

        # ─── 샘플 개수 계산 ───
        n_qt = int(round(self.qt_frac * n_total))
        n_train = int(round(self.train_ratio * n_total))
        n_rd = max(n_train - n_qt, 0)

        # ─── 분위 기반 샘플링 ───
        qt_idx = self._sample_quantile_weighted(vals, n_samples=n_qt, seed=seed_qt)
        remain = np.setdiff1d(idx_all, qt_idx)

        # ─── 랜덤 샘플링 ───
        rng = np.random.default_rng(seed_rd)
        rd_idx = rng.choice(remain, size=min(n_rd, len(remain)), replace=False)

        # ─── 테스트 세트 ───
        test_idx = np.setdiff1d(remain, rd_idx)
        train_idx = np.concatenate([qt_idx, rd_idx])

        # ─── 상세 로그 ───
        print("\n📊 [GCMCSampler: qt_then_rd]")
        print(f"   Total samples      : {n_total:,}")
        print(f"   Train/Test split   : {len(train_idx):,} / {len(test_idx):,} (target train={self.train_ratio:.2f})")
        print(f"   Quantile frac (γ_q): {self.qt_frac:.2f} → {len(qt_idx):,} samples ({len(qt_idx)/n_total:.2%} of total)")
        print(f"   Random samples     : {len(rd_idx):,} ({len(rd_idx)/n_total:.2%} of total)")
        print(f"   Remaining for test : {len(test_idx):,} ({len(test_idx)/n_total:.2%} of total)")
        print(f"   Seeds used         : qt={seed_qt}, rd={seed_rd}")
        print("───────────────────────────────────────────────")
        return {
            "train_idx": train_idx,
            "test_idx": test_idx,
            "train_qt_idx": qt_idx,
            "train_rd_idx": rd_idx,
        }


    # ───────────────────────────────────────────────
    def fit(self, df: pd.DataFrame) -> Dict[str, np.ndarray]:
        """메인 샘플링 실행"""
        df = df.reset_index(drop=True)
        n_total = len(df)

        seed_qt = self.seed_base + 1000
        seed_rd = self.seed_base + 2000
        seed_rd_simple = self.seed_base + 3000

        if self.sampler_type == "random_struct":
            result = self._split_random(n_total, seed_rd_simple)
        elif self.sampler_type == "random_with_input":
            result = self._split_random(n_total, seed_rd_simple)
        elif self.sampler_type == "qt_then_rd":
            result = self._split_qt_then_rd(df, seed_qt, seed_rd)
        else:
            raise ValueError(f"Unsupported sampler_type: {self.sampler_type}")
        return result

    # ───────────────────────────────────────────────
    def summary(self, result: Dict[str, np.ndarray], df: Optional[pd.DataFrame] = None) -> None:
        """샘플링 결과 통계 + 시각화"""
        n_tr = len(result["train_idx"])
        n_te = len(result["test_idx"])
        qt_n = len(result["train_qt_idx"]) if result["train_qt_idx"] is not None else 0
        rd_n = len(result["train_rd_idx"]) if result["train_rd_idx"] is not None else 0

        print(f"\n🧩 [GCMCSampler Summary]")
        print(f"   Sampler Type : {self.sampler_type}")
        print(f"   Train/Test   : {n_tr} / {n_te} (ratio={self.train_ratio:.2f})")
        if self.sampler_type == "qt_then_rd":
            print(f"   Quantile/Random : {qt_n} / {rd_n}")
            print(f"   Quantile Col: {self.qt_col}")
            print(f"   use_log={self.use_log}, n_bins={self.n_bins}, gamma={self.gamma}")
        print(f"   Seed Base: {self.seed_base}\n")

        # ---- 시각화 ----
        if df is None or self.qt_col is None or self.qt_col not in df.columns:
            return

        vals = df[self.qt_col].astype(float).values
        if self.use_log:
            vals = np.log10(np.clip(vals, a_min=self.log_eps, a_max=None))

        plt.figure(figsize=(8,5))
        plt.hist(vals, bins=self.n_bins, alpha=0.4, color="gray", label="All data")

        if result["train_qt_idx"] is not None and len(result["train_qt_idx"]) > 0:
            qt_vals = vals[result["train_qt_idx"]]
            plt.hist(qt_vals, bins=self.n_bins, alpha=0.6, color="orange", label="Quantile-sampled")

        if result["train_rd_idx"] is not None and len(result["train_rd_idx"]) > 0:
            rd_vals = vals[result["train_rd_idx"]]
            plt.hist(rd_vals, bins=self.n_bins, alpha=0.6, color="blue", label="Random-sampled")

        plt.xlabel(f"log10({self.qt_col})" if self.use_log else self.qt_col)
        plt.ylabel("Count")
        plt.title(f"Sampling Distribution — {self.qt_col}")
        plt.legend()
        plt.tight_layout()

        if self.outdir:
            import os
            os.makedirs(self.outdir, exist_ok=True)
            out_path = f"{self.outdir}/sampling_hist_{self.qt_col}.png"
            plt.savefig(out_path, dpi=250)
            print(f"📊 Plot saved → {out_path}")
        else:
            plt.show()
        plt.close()
