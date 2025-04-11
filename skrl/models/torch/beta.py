from typing import Any, Mapping, Tuple, Union

import gymnasium

import torch
from torch.distributions import Beta


# speed up distribution construction by disabling checking
Beta.set_default_validate_args(False)


class BetaMixin:
    def __init__(
        self,
        scaler: float = 1.0,
        clip_actions: bool = False,
        reduction: str = "sum",
        role: str = "",
    ) -> None:
        """Beta mixin model (stochastic model)
        """
        self._b_clip_actions = clip_actions and isinstance(self.action_space, gymnasium.Space)

        if self._b_clip_actions:
            self._b_clip_actions_min = torch.tensor(self.action_space.low, device=self.device, dtype=torch.float32)
            self._b_clip_actions_max = torch.tensor(self.action_space.high, device=self.device, dtype=torch.float32)

        self._b_alpha = None
        self._b_beta = None
        self._b_num_samples = None
        self._b_distribution = None

        if reduction not in ["mean", "sum", "prod", "none"]:
            raise ValueError("reduction must be one of 'mean', 'sum', 'prod' or 'none'")
        self._b_reduction = (
            torch.mean
            if reduction == "mean"
            else torch.sum if reduction == "sum" else torch.prod if reduction == "prod" else None
        )

    def act(
        self, inputs: Mapping[str, Union[torch.Tensor, Any]], role: str = ""
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None], Mapping[str, Union[torch.Tensor, Any]]]:
        """Act stochastically in response to the state of the environment

        :param inputs: Model inputs. The most common keys are:

                       - ``"states"``: state of the environment used to make the decision
                       - ``"taken_actions"``: actions taken by the policy for the given states
        :type inputs: dict where the values are typically torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        :return: Model output. The first component is the action to be taken by the agent.
                 The second component is the log of the probability density function.
                 The third component is a dictionary containing the mean actions ``"mean_actions"``
                 and extra output values
        :rtype: tuple of torch.Tensor, torch.Tensor or None, and dict

        Example::

            >>> # given a batch of sample states with shape (4096, 60)
            >>> actions, log_prob, outputs = model.act({"states": states})
            >>> print(actions.shape, log_prob.shape, outputs["mean_actions"].shape)
            torch.Size([4096, 8]) torch.Size([4096, 1]) torch.Size([4096, 8])
        """
        # map from states/observations to mean actions and log standard deviations
        log_alpha, log_beta, outputs = self.compute(inputs, role)
        alpha = log_alpha.exp()
        beta = log_beta.exp()

        self._b_alpha = alpha
        self._b_beta = beta
        self._b_num_samples = alpha.shape[0]

        # distribution
        self._b_distribution = Beta(alpha, beta)

        # sample using the reparameterization trick
        actions = self._b_distribution.rsample()

        # clip actions
        if self._b_clip_actions:
            actions = torch.clamp(actions, min=self._b_clip_actions_min, max=self._b_clip_actions_max)

        # log of the probability density function
        log_prob = self._b_distribution.log_prob(inputs.get("taken_actions", actions))
        if self._b_reduction is not None:
            log_prob = self._b_reduction(log_prob, dim=-1)
        if log_prob.dim() != actions.dim():
            log_prob = log_prob.unsqueeze(-1)

        outputs["alpha"] = alpha
        outputs["beta"] = beta

        scaled_actions = 2.0 * actions - 1.0
        return scaled_actions, log_prob, outputs

    def get_entropy(self, role: str = "") -> torch.Tensor:
        """Compute and return the entropy of the model

        :return: Entropy of the model
        :rtype: torch.Tensor
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> entropy = model.get_entropy()
            >>> print(entropy.shape)
            torch.Size([4096, 8])
        """
        if self._b_distribution is None:
            return torch.tensor(0.0, device=self.device)
        return self._b_distribution.entropy().to(self.device)

    def distribution(self, role: str = "") -> torch.distributions.Beta:
        """Get the current distribution of the model

        :return: Distribution of the model
        :rtype: torch.distributions.Normal
        :param role: Role play by the model (default: ``""``)
        :type role: str, optional

        Example::

            >>> distribution = model.distribution()
            >>> print(distribution)
            Normal(loc: torch.Size([4096, 8]), scale: torch.Size([4096, 8]))
        """
        return self._b_distribution

