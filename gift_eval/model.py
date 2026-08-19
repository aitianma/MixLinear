"""
MixLinear core forecasting module.

This is the architecture from the official MixLinear repository
(https://github.com/aitianma/MixLinear, models/MixLinear.py), lightly
cleaned up (debug prints removed, dtype hardened) for use as a plain
nn.Module inside the GIFT-Eval GluonTS wrapper. The math is unchanged.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixLinearCore(nn.Module):
    """Channel-independent dual-domain (time + frequency) linear forecaster.

    Input:  x of shape (batch, seq_len, enc_in)
    Output: y of shape (batch, pred_len, enc_in)

    Constraints (enforced by the caller, not this module):
      - seq_len must be an exact multiple of period_len
      - lpf must be <= ceil(seq_len / period_len)
    """

    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        enc_in: int,
        period_len: int,
        lpf: int,
        alpha: float,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.enc_in = enc_in
        self.period_len = period_len
        self.kernel = period_len
        self.lpf = lpf
        self.alpha = alpha

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)

        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seq_len / self.period_len))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.pred_len / self.period_len))

        # TLinear (time domain)
        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=self.kernel + 1,
            stride=1,
            padding=int(self.kernel / 2),
            padding_mode="zeros",
            bias=False,
        )

        # FLinear (frequency domain)
        self.FLinear1 = nn.Linear(self.lpf, 2, bias=False).to(torch.cfloat)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False).to(torch.cfloat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        # normalization and permute     b,s,c -> b,c,s
        seq_mean = torch.mean(x, dim=1).unsqueeze(1)
        x = (x - seq_mean).permute(0, 2, 1)
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x

        # ->b,e,w,n
        x = x.reshape(batch_size, self.enc_in, -1, self.period_len).permute(0, 1, 3, 2)

        # ---- Time domain ----
        x_o = F.pad(x, (0, self.sqrt_seg_num_x ** 2 - x.shape[-1], 0, 0, 0, 0))
        x_o = x_o.reshape(batch_size, self.enc_in, self.period_len, self.sqrt_seg_num_x, self.sqrt_seg_num_x)

        x_o = self.TLinear1(x_o).permute(0, 1, 2, 4, 3)
        x_t = self.TLinear2(x_o).permute(0, 1, 2, 4, 3)

        x_t = (
            x_t.reshape(batch_size, self.enc_in, self.period_len, -1)
            .permute(0, 1, 3, 2)
            .reshape(batch_size, self.enc_in, -1)
            .permute(0, 2, 1)
        )

        # ---- Frequency domain ----
        x_fft = torch.fft.fft(x, dim=3)[:, :, :, : self.lpf]
        x_fft = self.FLinear1(x_fft)
        x_fft = self.FLinear2(x_fft).reshape(batch_size, self.enc_in, self.period_len, -1)

        x_rfft = torch.fft.ifft(x_fft, dim=3).float()
        x_f = x_rfft.permute(0, 1, 3, 2).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

        # ---- Mix ----
        x_t = x_t[:, : self.pred_len, :]
        x_f = x_f[:, : self.pred_len, :]
        x = x_t * self.alpha + seq_mean + x_f * (1 - self.alpha)

        return x[:, : self.pred_len, :]
