import torch
import torch.nn as nn

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model

from .layers import HyperNormal, HyperLinear, HyperEmbedder, HyperLERPBlock


class SimbaV2Actor(GaussianMixin, Model):
    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

        self.embedder = HyperEmbedder(
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        self.predictor = HyperNormal(
            hidden_dim=hidden_dim,
            out_dim=self.num_actions,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def compute(self, inputs, role):
        x = inputs["states"]
        x = self.embedder(x)
        x = self.encoder(x)
        mean, log_std = self.predictor(x)

        log_std = self._log_std_min + (self._log_std_max - self._log_std_min) * 0.5 * (
            1 + torch.tanh(log_std)
        )
        mean = torch.tanh(mean)

        return mean, log_std, {}


class SimbaV2LinearCritic(DeterministicMixin, Model):
    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        observation_space,
        action_space,
        device,
        clip_actions=False,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)
        self.embedder = HyperEmbedder(
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )

        self.predictor = HyperLinear(
            hidden_dim=hidden_dim,
            out_dim=1,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        x = self.embedder(x)
        x = self.encoder(x)
        q = self.predictor(x)

        return q, {}


class SimbaV2CategoricalCritic(DeterministicMixin, Model):
    def __init__(
        self,
        hidden_dim: int,
        num_blocks: int,
        num_bins: int,
        scaler_init: float,
        scaler_scale: float,
        alpha_init: float,
        alpha_scale: float,
        c_shift: float,
        observation_space,
        action_space,
        device,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False)
        self.embedder = HyperEmbedder(
            hidden_dim=hidden_dim,
            scaler_init=scaler_init,
            scaler_scale=scaler_scale,
            c_shift=c_shift,
        )

        self.encoder = nn.Sequential(
            *[
                HyperLERPBlock(
                    hidden_dim=hidden_dim,
                    scaler_init=scaler_init,
                    scaler_scale=scaler_scale,
                    alpha_init=alpha_init,
                    alpha_scale=alpha_scale,
                )
                for _ in range(num_blocks)
            ]
        )
        self.predictor = HyperLinear(
            hidden_dim=hidden_dim,
            out_dim=num_bins,
            scaler_init=1.0,
            scaler_scale=1.0,
        )

    def compute(self, inputs, role):
        x = torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)
        x = self.embedder(x)
        x = self.encoder(x)
        x = self.predictor(x)

        return x, {}
