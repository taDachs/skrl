import gymnasium as gym

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config
from skrl.utils.spaces.torch import compute_space_size


def l2normalize_model(model: nn.Module):
    # Iterate over all modules in the model
    for name, module in model.named_modules():
        # Check if the module is a linear layer
        if "hyper_w" in name:
            # Normalize the weights using L2 norm
            with torch.no_grad():
                weight = module.weight
                normalized_weight = F.normalize(weight, p=2, dim=1)
                module.weight.copy_(normalized_weight)


class NumpyRunningMeanStd:
    """Tracks the mean, variance and count of values."""

    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), dtype=np.float64):
        """Tracks the mean, variance and count of values."""
        self.mean = np.zeros(shape, dtype=dtype)
        self.var = np.ones(shape, dtype=dtype)
        self.count = epsilon

    def update(self, x):
        """Updates the mean, var and count from a batch of samples."""
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Updates from batch mean, variance and count moments."""
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count
        )


def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    """Updates the mean, var and count using the previous mean, var, count and batch values."""
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return new_mean, new_var, new_count


class TorchRunningMeanStd(nn.Module):
    """
    A drop-in 'nn.Module' style class for online mean-variance normalization,
    implemented similarly to RunningStandardScaler, but named TorchRunningMeanStd.
    """

    def __init__(
        self, size, epsilon: float = 1e-8, clip_threshold: float = 5.0, device: str = None
    ):
        """
        :param size: Size of the input data
        :param epsilon: Small number to avoid division by zero
        :param clip_threshold: Threshold to clip the standardized data
        :param device: Device to store buffers. Defaults to GPU if available, else CPU.
        """
        super().__init__()

        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        # resolve and store the device
        self.device = config.torch.parse_device(device)

        # if 'size' is a gymnasium space, compute the flattened dimension
        size = compute_space_size(size, occupied_size=True)

        # register buffers to properly move them between CPU/GPU
        self.register_buffer(
            "running_mean", torch.zeros(size, dtype=torch.float64, device=self.device)
        )
        self.register_buffer(
            "running_variance", torch.ones(size, dtype=torch.float64, device=self.device)
        )
        self.register_buffer(
            "current_count", torch.ones((), dtype=torch.float64, device=self.device)
        )

    @torch.no_grad()
    def _parallel_variance(
        self, input_mean: torch.Tensor, input_var: torch.Tensor, input_count: int
    ):
        """
        Update running mean/variance using the parallel algorithm for batch statistics.

        Reference: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
        https://github.com/SonyResearch/simba/blob/master/scale_rl/agents/wrappers/utils.py def update_mean_var_count_from_moments
        """
        delta = input_mean - self.running_mean
        total_count = self.current_count + input_count

        # M2 is the sum of squared differences from the (old) mean
        M2 = (
            (self.running_variance * self.current_count)
            + (input_var * input_count)
            + (delta**2) * self.current_count * input_count / total_count
        )

        # update internal variables
        self.running_mean = self.running_mean + delta * input_count / total_count
        self.running_variance = M2 / total_count
        self.current_count = total_count

    def _compute(self, x: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        """
        Perform the actual forward computation:
         - If `train=True`, update running stats
         - If `inverse=True`, unscale (go from standardized -> original)
         - Else, standardize (original -> standardized)
        """
        # update running stats if in "training" mode
        if train:
            # shape can be (batch, obs_dim) or (num_envs, num_steps, obs_dim), etc.
            # sum up total elements for parallel variance
            if x.dim() > 2:
                # e.g. (N, T, obs_dim)
                input_count = x.size(0) * x.size(1)
                x_mean = torch.mean(x, dim=(0, 1))
                x_var = torch.var(x, dim=(0, 1), unbiased=False)
            else:
                # e.g. (N, obs_dim)
                input_count = x.size(0)
                x_mean = torch.mean(x, dim=0)
                x_var = torch.var(x, dim=0, unbiased=False)

            self._parallel_variance(x_mean, x_var, input_count)

        # if we are in "inverse" mode, scale back from standardized to original
        if inverse:
            return (
                torch.sqrt(self.running_variance.float())
                * torch.clamp(x, min=-self.clip_threshold, max=self.clip_threshold)
                + self.running_mean.float()
            )

        # otherwise, standardize from original to standardized
        return torch.clamp(
            (x - self.running_mean.float())
            / (torch.sqrt(self.running_variance.float()) + self.epsilon),
            min=-self.clip_threshold,
            max=self.clip_threshold,
        )

    def forward(
        self, x: torch.Tensor, train: bool = False, inverse: bool = False, no_grad: bool = True
    ) -> torch.Tensor:
        """
        Public entry point for the module, mirroring 'RunningStandardScaler'.

        - no_grad: whether to turn off gradients for the standardization step
        - train:   whether to update running statistics
        - inverse: whether to invert standardization
        """
        if no_grad:
            with torch.no_grad():
                return self._compute(x, train=train, inverse=inverse)
        else:
            return self._compute(x, train=train, inverse=inverse)


class NormalizeReward(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(
        self,
        env: gym.Env,
        gamma: float = 0.99,
        g_max: float = 10.0,
        epsilon: float = 1e-8,
    ):
        """This wrapper will normalize immediate rewards s.t. their exponential moving average has
        an approximately fixed variance.

        Args:
            env (env): The environment to apply the wrapper
            epsilon (float): A stability parameter
            gamma (float): The discount factor that is used in the exponential moving average.
        """
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)

        self.return_rms = NumpyRunningMeanStd(shape=())
        self.reward: np.array = np.array([0.0])
        self.gamma = gamma
        self.epsilon = epsilon
        self.g_max = g_max
        self._update_running_mean = True
        self.max_return = 0.0

    @property
    def update_running_mean(self) -> bool:
        """Property to freeze/continue the running mean calculation of the reward statistics."""
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        """Sets the property to freeze/continue the running mean calculation of the reward statistics."""
        self._update_running_mean = setting

    def step(self, action) -> tuple:
        """Steps through the environment, normalizing the reward returned."""
        obs, reward, terminated, truncated, info = super().step(action)

        self.reward = self.reward * self.gamma + float(reward)
        if self._update_running_mean:
            self.return_rms.update(self.reward)

        self.max_return = max(self.max_return, max(abs(self.reward)))

        var_denominator = np.sqrt(self.return_rms.var + self.epsilon)
        min_required_denominator = self.max_return / self.g_max
        denominator = max(var_denominator, min_required_denominator)

        normalized_reward = reward / denominator

        if terminated or truncated:
            self.reward = np.array([0.0])

        return obs, normalized_reward, terminated, truncated, info
