import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class Scaler(nn.Module):
    def __init__(self, dim: int, init: float = 1.0, scale: float = 1.0):
        super().__init__()
        self.scaler = nn.Parameter(torch.ones(dim) * scale, requires_grad=True)
        self.forward_scaler = init / scale

    def forward(self, x):
        return self.scaler * self.forward_scaler * x


class HyperDense(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.hyper_w = nn.LazyLinear(out_dim, bias=False)
        self.hyper_w.reset_parameters = lambda: nn.init.orthogonal_(self.hyper_w.weight, gain=1.0)

    def forward(self, x):
        return self.hyper_w(x)


class HyperMLP(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float,
        scaler_scale: float,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.eps = eps

        self.w1 = HyperDense(hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(out_dim)

    def forward(self, x):
        x = self.w1(x)
        x = self.scaler(x)
        x = F.relu(x) + self.eps
        x = self.w2(x)
        x = F.normalize(x, p=2, dim=-1)
        return x


class HyperEmbedder(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float = None,
        scaler_scale: float = None,
        c_shift: float = 3.0,
    ):
        super().__init__()
        if not scaler_init:
            scaler_init = math.sqrt(2 / hidden_dim)
        if not scaler_scale:
            scaler_scale = math.sqrt(2 / hidden_dim)
        self.c_shift = c_shift

        self.w = HyperDense(hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)

    def forward(self, x):
        new_axis = torch.ones(x.shape[:-1] + (1,), device=x.device) * self.c_shift

        x = torch.cat([x, new_axis], dim=-1)
        x = F.normalize(x, p=2, dim=-1)
        x = self.w(x)
        x = self.scaler(x)
        x = F.normalize(x, p=2, dim=-1)

        return x


class HyperLERPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        scaler_init: float = None,
        scaler_scale: float = None,
        alpha_init: float = None,
        alpha_scale: float = None,
        expansion: int = 4,
    ):
        super().__init__()
        if not scaler_init:
            scaler_init = math.sqrt(2 / hidden_dim)
        if not scaler_scale:
            scaler_scale = math.sqrt(2 / hidden_dim)
        if not alpha_init:
            alpha_init = 1 / (hidden_dim + 1)
        if not alpha_scale:
            alpha_scale = 1 / math.sqrt(hidden_dim)

        self.mlp = HyperMLP(
            hidden_dim * expansion,
            hidden_dim,
            scaler_init / torch.math.sqrt(expansion),
            scaler_scale / torch.math.sqrt(expansion),
        )

        self.alpha_scaler = Scaler(hidden_dim, alpha_init, alpha_scale)

    def forward(self, x):
        residual = x
        x = self.mlp(x)
        x = residual + self.alpha_scaler(x - residual)
        x = F.normalize(x, p=2, dim=-1)

        return x

class HyperNormal(nn.Module):
    def __init__(
        self,
        out_dim: int,
        hidden_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ):
        super().__init__()

        self.mean_w1 = HyperDense(hidden_dim)
        self.mean_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.mean_w2 = HyperDense(out_dim)
        self.mean_bias = nn.Parameter(torch.zeros(out_dim), requires_grad=True)

        self.std_w1 = HyperDense(hidden_dim)
        self.std_scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.std_w2 = HyperDense(out_dim)
        self.std_bias = nn.Parameter(torch.zeros(out_dim), requires_grad=True)

    def forward(self, x):
        mean = self.mean_w1(x)
        mean = self.mean_scaler(mean)
        mean = self.mean_w2(mean) + self.mean_bias

        log_std = self.std_w1(x)
        log_std = self.std_scaler(log_std)
        log_std = self.std_w2(log_std) + self.std_bias

        return mean, log_std


class HyperLinear(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        out_dim: int,
        scaler_init: float = 1.0,
        scaler_scale: float = 1.0,
    ):
        super().__init__()

        self.w1 = HyperDense(hidden_dim)
        self.scaler = Scaler(hidden_dim, scaler_init, scaler_scale)
        self.w2 = HyperDense(out_dim)

        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        x = self.w1(x)
        x = self.scaler(x)
        x = self.w2(x) + self.bias

        return x

class TanhNormalSquasher(nn.Module):
    def __init__(
        self,
        scaler: float = 1.0,
        min_log_std=-20,
        max_log_std=2,
    ):
        super().__init__()

        self.scaler = scaler
        self._log_std_min = min_log_std
        self._log_std_max = max_log_std

    def forward(self, x):
        mean, log_std = x
        log_std = self._log_std_min + (self._log_std_max - self._log_std_min) * 0.5 * (
            1 + torch.tanh(log_std)
        )
        mean = self.scaler * torch.tanh(mean)
        return mean, log_std
