"""Quantum ML model: variational quantum regressor on a state-vector SIMULATOR (FR-08)."""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from ifap_common.config import env_int
from ifap_common.errors import AppError
from ifap_common.modelservice import BaseModelImpl

DEVICE = "default.qubit"


def _circuit(qml, torch, n_qubits, layers):
    dev = qml.device(DEVICE, wires=n_qubits)

    @qml.qnode(dev, interface="torch", diff_method="backprop")
    def circuit(x, w):
        qml.AngleEmbedding(x, wires=range(n_qubits), rotation="Y")
        qml.StronglyEntanglingLayers(w, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(0))
    return circuit


class VariationalRegressor(BaseModelImpl):
    name, label, tasks = "qml_vqr", "Variational quantum regressor (simulator)", ["VOLATILITY", "OPTION_PRICE"]
    description = ("Angle-encoded variational circuit (StronglyEntanglingLayers), <Z0> read-out with a trainable "
                   "affine map to the target. Runs on PennyLane's default.qubit SIMULATOR; no quantum hardware or "
                   "quantum advantage is implied.")
    default_config = {"n_qubits": 4, "layers": 3, "epochs": 120, "lr": 0.1, "max_train_rows": 400}
    bounds = {"n_qubits": (1, 64), "layers": (1, 6), "epochs": (1, 300), "lr": (1e-4, 1.0),
              "max_train_rows": (20, 5000)}

    def validate_config(self, config):
        c = super().validate_config(config)
        limit = env_int("IFAP_MAX_QUBITS", 8)
        if int(c["n_qubits"]) > limit:
            raise AppError("QUBIT_LIMIT_EXCEEDED",
                           f"Requested {int(c['n_qubits'])} qubits; the simulator limit is {limit}.",
                           fields=[{"field": "n_qubits"}])
        return c

    def fit(self, fs, config, seed, out_dir: Path):
        import pennylane as qml
        import torch
        torch.set_num_threads(2)
        torch.manual_seed(seed)
        nq, layers = int(config["n_qubits"]), int(config["layers"])
        if nq > len(fs.names):
            raise AppError("CONFIG_INVALID", "n_qubits exceeds the number of available features.")
        tr0, tr1 = fs.split_range("train")
        raw = fs.a["X_raw"][tr0:tr1]
        X = fs.scale(raw)[:, :nq]
        y = fs.a["y"][tr0:tr1] / self.target_scale(fs, raw)
        rng = np.random.default_rng(seed)
        if len(y) > int(config["max_train_rows"]):  # seeded, order-preserving subsample bounds simulator cost
            keep = np.sort(rng.choice(len(y), int(config["max_train_rows"]), replace=False))
            X, y = X[keep], y[keep]
        ym, ys = float(y.mean()), float(y.std() or 1.0)
        xt = torch.tensor(np.arctan(X), dtype=torch.float64) * 2.0   # squash to (-pi, pi)
        yt = torch.tensor((y - ym) / ys, dtype=torch.float64)
        circuit = _circuit(qml, torch, nq, layers)
        shape = qml.StronglyEntanglingLayers.shape(n_layers=layers, n_wires=nq)
        w = torch.tensor(rng.normal(0, 0.1, shape), dtype=torch.float64, requires_grad=True)
        a = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        b = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([w, a, b], lr=float(config["lr"]))
        t0, losses = time.time(), []
        for _ in range(int(config["epochs"])):
            opt.zero_grad()
            pred = a * circuit(xt, w) + b
            loss = torch.mean((pred - yt) ** 2)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        torch.save({"w": w.detach(), "a": a.detach(), "b": b.detach(), "nq": nq, "layers": layers,
                    "ym": ym, "ys": ys}, out_dir / "model.pt")
        return {"seed": seed, "simulator": DEVICE, "device": "state-vector simulator (classical CPU)",
                "nQubits": nq, "layers": layers, "nTrainRows": int(len(y)), "finalTrainLoss": losses[-1],
                "trainSeconds": round(time.time() - t0, 2), "inputs": fs.names[:nq]}

    def predict(self, fs, frame, out_dir: Path):
        import pennylane as qml
        import torch
        torch.set_num_threads(2)
        ck = torch.load(out_dir / "model.pt", weights_only=True)
        nq = int(ck["nq"])
        circuit = _circuit(qml, torch, nq, int(ck["layers"]))
        xt = torch.tensor(np.arctan(frame.X[:, :nq]), dtype=torch.float64) * 2.0
        with torch.no_grad():
            out = (ck["a"] * circuit(xt, ck["w"]) + ck["b"]).numpy().astype(float)
        return (out * ck["ys"] + ck["ym"]) * self.target_scale(fs, frame.X_raw)
