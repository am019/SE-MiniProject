"""Deep-learning model: LSTM sequence regressor (FR-07)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ifap_common.modelservice import BaseModelImpl


def _build_net(torch, d_dyn, d_static, hidden):
    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(d_dyn, hidden, batch_first=True)
            self.head = torch.nn.Sequential(torch.nn.Linear(hidden + d_static, 16), torch.nn.ReLU(),
                                            torch.nn.Linear(16, 1))

        def forward(self, seq, static):
            _, (h, _) = self.lstm(seq)
            return self.head(torch.cat([h[-1], static], dim=1)).squeeze(1)
    return Net()


class Lstm(BaseModelImpl):
    name, label, tasks = "lstm", "LSTM", ["VOLATILITY", "OPTION_PRICE"]
    description = ("LSTM over the recent history of daily features (+ static contract features for option "
                   "prices). Early stopping on the chronologically last part of the TRAIN rows; best weights restored.")
    default_config = {"seq_len": 20, "hidden": 32, "epochs": 40, "batch_size": 64, "lr": 0.003,
                      "patience": 6, "holdout_fraction": 0.15}
    bounds = {"seq_len": (2, 120), "hidden": (4, 256), "epochs": (1, 500), "batch_size": (8, 1024),
              "lr": (1e-5, 0.5), "patience": (1, 100), "holdout_fraction": (0.05, 0.4)}

    def fit(self, fs, config, seed, out_dir: Path):
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.use_deterministic_algorithms(True)
        L = int(config["seq_len"])
        tr0, tr1 = fs.split_range("train")
        raw = fs.a["X_raw"][tr0:tr1]
        from ifap_common.featureset import Frame
        frame = Frame([], raw, fs.scale(raw), fs.a["y"][tr0:tr1], None, fs.a["seq_idx"][tr0:tr1])
        seq = torch.tensor(fs.seq_tensor(frame, L), dtype=torch.float32)
        st = torch.tensor(fs.static_matrix(frame), dtype=torch.float32)
        y = fs.a["y"][tr0:tr1] / self.target_scale(fs, raw)
        ym, ys = float(y.mean()), float(y.std() or 1.0)
        yt = torch.tensor((y - ym) / ys, dtype=torch.float32)
        n = len(yt)
        n_hold = max(5, int(n * float(config["holdout_fraction"])))
        n_fit = n - n_hold
        net = _build_net(torch, seq.shape[2], st.shape[1], int(config["hidden"]))
        opt = torch.optim.Adam(net.parameters(), lr=float(config["lr"]))
        lossf = torch.nn.MSELoss()
        g = torch.Generator().manual_seed(seed)
        best, best_state, bad, epochs_run = 1e18, None, 0, 0
        bs = int(config["batch_size"])
        t0 = time.time()
        for ep in range(int(config["epochs"])):
            net.train()
            perm = torch.randperm(n_fit, generator=g)
            for i in range(0, n_fit, bs):
                b = perm[i:i + bs]
                opt.zero_grad()
                loss = lossf(net(seq[b], st[b]), yt[b])
                loss.backward()
                opt.step()
            net.eval()
            with torch.no_grad():
                hl = float(lossf(net(seq[n_fit:], st[n_fit:]), yt[n_fit:]))
            epochs_run = ep + 1
            if hl < best - 1e-4:
                best, bad = hl, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= int(config["patience"]):
                    break
        net.load_state_dict(best_state)
        torch.save({"state": net.state_dict(), "cfg": {k: config[k] for k in ("seq_len", "hidden")},
                    "d_dyn": seq.shape[2], "d_static": st.shape[1], "ym": ym, "ys": ys}, out_dir / "model.pt")
        return {"seed": seed, "epochsRun": epochs_run, "maxEpochs": int(config["epochs"]),
                "earlyStopping": f"patience {config['patience']} epochs on the last {n_hold} train rows "
                                 f"(min delta 1e-4); best weights restored",
                "bestHoldoutLoss": best, "trainSeconds": round(time.time() - t0, 2), "device": "cpu"}

    def predict(self, fs, frame, out_dir: Path):
        import torch
        torch.set_num_threads(2)
        ck = torch.load(out_dir / "model.pt", weights_only=True)  # hash verified by the service first
        net = _build_net(torch, ck["d_dyn"], ck["d_static"], int(ck["cfg"]["hidden"]))
        net.load_state_dict(ck["state"])
        net.eval()
        seq = torch.tensor(fs.seq_tensor(frame, int(ck["cfg"]["seq_len"])), dtype=torch.float32)
        st = torch.tensor(fs.static_matrix(frame), dtype=torch.float32)
        with torch.no_grad():
            out = net(seq, st).numpy().astype(float)
        return (out * ck["ys"] + ck["ym"]) * self.target_scale(fs, frame.X_raw)
