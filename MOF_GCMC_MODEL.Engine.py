# -*- coding: utf-8 -*-
# ActiveLearning/AL_Engine.py
import os
import gc
import datetime
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ActiveLearning.AL_Sampler import ActiveSampler
from ActiveLearning.AL_Trainer import ActiveTrainer
from sklearn.preprocessing import StandardScaler
from torch import nn, optim


# ── Dataset / Model definitions ───────────────────────────
class GCMCDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y_log, LOW_log):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y_log, dtype=torch.float32).unsqueeze(1)
        self.LOW = torch.tensor(LOW_log, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.Y[i], self.LOW[i]


class ResidualBlock(nn.Module):
    def __init__(self, dim, drop, activation):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), activation, nn.Dropout(drop),
            nn.Linear(dim, dim), activation, nn.Dropout(drop),
        )
    def forward(self, x): return x + self.block(x)


class GCMCModel(nn.Module):
    def __init__(self, dim, hidden1, hidden2, drop, activation):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(dim, hidden1), activation)
        self.res1 = ResidualBlock(hidden1, drop, activation)
        self.mid = nn.Sequential(nn.Linear(hidden1, hidden2), activation)
        self.res2 = ResidualBlock(hidden2, drop, activation)
        self.out = nn.Linear(hidden2, 1)
    def forward(self, x):
        x = self.input(x)
        x = self.res1(x)
        x = self.mid(x)
        x = self.res2(x)
        return self.out(x)


# ── Active Learning Engine ────────────────────────────────
class ActiveLearningEngine:
    def __init__(self, df, args):
        self.df = df
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._prepare_data()

        self.sampler = ActiveSampler(args.rd_frac, args.qt_frac, args.num_bins)
        self.metrics = []

    def _prepare_data(self):
        X_all = self.df.iloc[:, 1:-1].values.astype(np.float32)
        Y_all = self.df.iloc[:, -1].values.astype(np.float32)
        LOW_all = X_all[:, -1]
        self.LOW_log = np.log(LOW_all)
        self.Y_log = np.log(Y_all)
        X_all[:, -1] = self.LOW_log
        self.scaler_X = StandardScaler().fit(X_all) if self.args.x_scale else None
        self.X_scaled = self.scaler_X.transform(X_all) if self.args.x_scale else X_all
        self.Y_all = Y_all

    def _make_model(self):
        act_map = {"relu": nn.ReLU(), "silu": nn.SiLU(), "gelu": nn.GELU(), "mish": nn.Mish()}
        activation = act_map.get(self.args.activation, nn.ReLU())
        model = GCMCModel(self.X_scaled.shape[1], self.args.hidden_dim1,
                          self.args.hidden_dim2, self.args.dropout_rate, activation)
        optimizer = optim.Adam(model.parameters(), lr=self.args.lr)
        trainer = ActiveTrainer(model, optimizer, nn.MSELoss(),
                                epochs=self.args.epochs, patience=self.args.patience,
                                device=self.device)
        return trainer

    def run(self):
        idx_all = np.arange(len(self.X_scaled))
        n_init = int(self.args.initial_ratio * len(idx_all))
        idx_labeled = self.sampler.quantile_sampling(self.LOW_log, idx_all, n_init)
        idx_unlabeled = np.setdiff1d(idx_all, idx_labeled)
        n_target = int(self.args.target_ratio * len(idx_all))
        max_iters = (n_target - len(idx_labeled)) // self.args.samples_per_iter + 1

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = f"ALRUN_{timestamp}"
        os.makedirs(out_dir, exist_ok=True)

        for it in range(max_iters):
            print(f"\n[Iter {it}] Labeled={len(idx_labeled)} / Target={n_target}")
            trainer = self._make_model()

            # Train
            d_train = GCMCDataset(self.X_scaled[idx_labeled], self.Y_log[idx_labeled], self.LOW_log[idx_labeled])
            dl_train = DataLoader(d_train, batch_size=self.args.batch_size, shuffle=True)
            trainer.fit(dl_train, dl_train)

            # Evaluate (on unlabeled pool)
            X_rem = torch.tensor(self.X_scaled[idx_unlabeled]).float().to(self.device)
            y_true = self.Y_all[idx_unlabeled]
            y_pred_log, unc = trainer.predict_with_uncertainty(X_rem, self.args.mcd_n)
            y_pred = np.exp(np.clip(y_pred_log, -20, 20))
            metrics = trainer.evaluate(y_true, y_pred)
            print(f"→ R2={metrics['r2']:.4f}, MAE={metrics['mae']:.4f}, MSE={metrics['mse']:.4f}")
            self.metrics.append({"iter": it, "n_labeled": len(idx_labeled), **metrics})
            pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(f"{out_dir}/pred_iter{it}.csv", index=False)

            if len(idx_labeled) >= n_target:
                break

            # Select new samples
            new_idx = self.sampler.sample_next(idx_unlabeled, self.LOW_log, unc, self.args.samples_per_iter)
            idx_labeled = np.concatenate([idx_labeled, new_idx])
            idx_unlabeled = np.setdiff1d(idx_unlabeled, new_idx)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        pd.DataFrame(self.metrics).to_csv(f"{out_dir}/al_metrics.csv", index=False)
        print(f"\n✅ Finished. Results saved in {out_dir}/")


# ── Entry (for script run) ────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim1", type=int, default=128)
    parser.add_argument("--hidden_dim2", type=int, default=64)
    parser.add_argument("--dropout_rate", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--initial_ratio", type=float, default=0.01)
    parser.add_argument("--samples_per_iter", type=int, default=10)
    parser.add_argument("--target_ratio", type=float, default=0.3)
    parser.add_argument("--mcd_n", type=int, default=20)
    parser.add_argument("--num_bins", type=int, default=10)
    parser.add_argument("--x_scale", action="store_true")
    parser.add_argument("--rd_frac", type=float, default=0.3)
    parser.add_argument("--qt_frac", type=float, default=0.3)
    parser.add_argument("--activation", type=str, default="gelu")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    engine = ActiveLearningEngine(df, args)
    engine.run()
