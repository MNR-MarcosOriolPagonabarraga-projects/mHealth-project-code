r"""
PyTorch model for arousal prediction from preprocessed windows.

Expects tensors shaped like ``process.py`` exports:

- ``signals``: (batch, 2, 6000) — two EEG channels × 30 s @ 200 Hz
- ``psd``: (batch, 2, F) — Welch PSD bins (typically F ≈ 159 for 0.5–40 Hz)

Use ``signals`` alone by passing ``psd=None`` (spectral branch is skipped).

Install PyTorch::

    pip install -e '.[training]'

or ``pip install torch``.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional dependency
    torch = None  # type: ignore[misc, assignment]
    nn = None  # type: ignore[misc, assignment]
    F = None  # type: ignore[misc, assignment]

_TORCH_MISSING = torch is None


def _require_torch() -> None:
    if _TORCH_MISSING:
        raise ImportError(
            "This component requires PyTorch. Install with `pip install torch` "
            "or from the repo root: `pip install -e '.[training]'`."
        ) from None


# Matches default preprocessing output (Welch bins in band 0.5–40 Hz at fs=200).
DEFAULT_PSD_FREQ_BINS = 159
DEFAULT_SIGNAL_LEN = 6000


class PreprocessedWindowDataset:
    """Loads one ``*_preprocessed.npz`` (map-style: ``__len__`` / ``__getitem__``)."""

    def __init__(self, npz_path: str) -> None:
        data = dict(np.load(npz_path, allow_pickle=True))
        self.signals = data["signals"]  # (N, 2, L)
        self.psd = data["psd"]  # (N, 2, F)
        self.labels = data["labels"].astype(np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int) -> tuple:
        _require_torch()
        assert torch is not None
        x_t = torch.from_numpy(self.signals[idx].copy()).float()
        x_s = torch.from_numpy(self.psd[idx].copy()).float()
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x_t, x_s, y


if not _TORCH_MISSING:
    assert nn is not None and F is not None and torch is not None

    def _sep_conv_down(in_ch: int, out_ch: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    class _TemporalBranch(nn.Module):
        """1D CNN stack on (B, 2, L)."""

        def __init__(self, in_channels: int = 2, emb_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                _sep_conv_down(in_channels, 32),
                _sep_conv_down(32, 64),
                _sep_conv_down(64, emb_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.net(x)
            return h.mean(dim=-1)

    class _SpectralBranch(nn.Module):
        """1D CNN along frequency axis (B, 2, F) treated as two channels."""

        def __init__(self, freq_bins: int = DEFAULT_PSD_FREQ_BINS, emb_dim: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(2, 32, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm1d(32),
                nn.ReLU(inplace=True),
                nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.Conv1d(64, emb_dim, kernel_size=5, stride=2, padding=2, bias=False),
                nn.BatchNorm1d(emb_dim),
                nn.ReLU(inplace=True),
            )
            self.freq_bins = freq_bins

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            h = self.net(x)
            return h.mean(dim=-1)

    class SleepArousalNet(nn.Module):
        """
        Fused spectral + temporal CNN for binary classification (non-arousal vs arousal).

        logits shape: (batch, 2) for ``CrossEntropyLoss``.
        """

        def __init__(
            self,
            *,
            temporal_len: int = DEFAULT_SIGNAL_LEN,
            psd_bins: int = DEFAULT_PSD_FREQ_BINS,
            emb_dim: int = 128,
            dropout: float = 0.2,
            use_psd_branch: bool = True,
        ) -> None:
            super().__init__()
            self.use_psd_branch = use_psd_branch
            self.temporal = _TemporalBranch(2, emb_dim)
            self.spectral = _SpectralBranch(psd_bins, emb_dim) if use_psd_branch else None
            fused_in = emb_dim * 2 if use_psd_branch else emb_dim
            self.head = nn.Sequential(
                nn.Linear(fused_in, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, 2),
            )
            self._temporal_len = temporal_len
            self._psd_bins = psd_bins

        def forward(self, signals: torch.Tensor, psd: torch.Tensor | None = None) -> torch.Tensor:
            if signals.dim() != 3 or signals.size(1) != 2:
                raise ValueError(f"signals must be (N, 2, L); got {tuple(signals.shape)}")

            t_emb = self.temporal(signals)

            if not self.use_psd_branch:
                return self.head(t_emb)

            if psd is None:
                psd_dev = signals.device
                psd_dt = signals.dtype
                psd = torch.zeros(signals.size(0), 2, self._psd_bins, device=psd_dev, dtype=psd_dt)
            elif psd.dim() != 3 or psd.size(1) != 2:
                raise ValueError(f"psd must be (N, 2, F); got {tuple(psd.shape)}")

            assert self.spectral is not None
            s_emb = self.spectral(psd)
            x = torch.cat([t_emb, s_emb], dim=1)
            return self.head(x)

    def logits_to_proba_arousal(logits: torch.Tensor) -> torch.Tensor:
        """Class-1 probability for arousal."""
        return F.softmax(logits, dim=1)[:, 1]

else:

    class SleepArousalNet:  # type: ignore[no-redef]
        """Placeholder when PyTorch is not installed."""

        def __init__(self, *args: object, **kwargs: object) -> None:
            _require_torch()

    def logits_to_proba_arousal(logits: object) -> object:  # type: ignore[no-redef]
        _require_torch()
        raise AssertionError("unreachable")
