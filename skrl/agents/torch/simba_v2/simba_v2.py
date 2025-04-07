from typing import Any, Mapping, Optional, Tuple, Union

import copy
import itertools
import gymnasium
from packaging import version

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from skrl import config, logger
from skrl.agents.torch import Agent
from skrl.memories.torch import Memory
from skrl.models.torch import Model
from skrl.models.torch.deterministic import DeterministicMixin
from skrl.models.torch.gaussian import GaussianMixin

from .normalization import l2normalize_model


# fmt: off
# [start-config-dict-torch]
SIMBAV2_DEFAULT_CONFIG = {
    "gradient_steps": 1,            # gradient steps
    "batch_size": 64,               # training batch size

    "discount_factor": 0.99,        # discount factor (gamma)
    "polyak": 0.005,                # soft update hyperparameter (tau)

    "actor_learning_rate": 1e-3,    # actor learning rate
    "critic_learning_rate": 1e-3,   # critic learning rate
    "learning_rate_scheduler": None,        # learning rate scheduler class (see torch.optim.lr_scheduler)
    "learning_rate_scheduler_kwargs": {},   # learning rate scheduler's kwargs (e.g. {"step_size": 1e-3})

    "state_preprocessor": None,             # state preprocessor class (see skrl.resources.preprocessors)
    "state_preprocessor_kwargs": {},        # state preprocessor's kwargs (e.g. {"size": env.observation_space})

    "random_timesteps": 0,          # random exploration steps
    "learning_starts": 0,           # learning starts after this many steps

    "grad_norm_clip": 0,            # clipping coefficient for the norm of the gradients

    "learn_entropy": True,          # learn entropy
    "entropy_learning_rate": 1e-3,  # entropy learning rate
    "initial_entropy_value": 0.2,   # initial entropy value
    "target_entropy": None,         # target entropy

    "normalize_weights": True,
    "min_v": -10.0,  # min critic value
    "max_v": 10.0,  # max critic value
    "num_bins": 101,  # num of bins for critic
    "use_categorical_critic": True,

    "rewards_shaper": None,         # rewards shaping function: Callable(reward, timestep, timesteps) -> reward

    "mixed_precision": False,       # enable automatic mixed precision for higher performance

    "experiment": {
        "directory": "",            # experiment's parent directory
        "experiment_name": "",      # experiment name
        "write_interval": "auto",   # TensorBoard writing interval (timesteps)

        "checkpoint_interval": "auto",      # interval for checkpoints (timesteps)
        "store_separately": False,          # whether to store checkpoints separately

        "wandb": False,             # whether to use Weights & Biases
        "wandb_kwargs": {}          # wandb kwargs (see https://docs.wandb.ai/ref/python/init)
    }
}
# [end-config-dict-torch]
# fmt: on


def categorical_td_loss(
    pred_log_probs: torch.Tensor,  # (n, num_bins)
    target_log_probs: torch.Tensor,  # (n, num_bins)
    reward: torch.Tensor,  # (n, 1)
    done: torch.Tensor,  # (n,)
    actor_log_probs: torch.Tensor,  # (n,)
    entropy_coefficient: torch.Tensor,  # (1,)
    gamma: float,
    num_bins: int,
    min_v: float,
    max_v: float,
    device: torch.device,
) -> torch.Tensor:
    with torch.no_grad():
        actor_entropy = actor_log_probs * entropy_coefficient

        bin_values = torch.linspace(min_v, max_v, num_bins, device=device).reshape(1, -1)
        target_bin_values = reward + gamma * (bin_values - actor_entropy) * (1.0 - done)
        target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

        b = (target_bin_values - min_v) / (max_v - min_v) * (num_bins - 1)
        l = torch.floor(b)
        u = torch.ceil(b)

        l_mask = F.one_hot(l.reshape(-1).long(), num_classes=num_bins).reshape(
            -1, num_bins, num_bins
        )
        u_mask = F.one_hot(u.reshape(-1).long(), num_classes=num_bins).reshape(
            -1, num_bins, num_bins
        )

        target_probs = torch.exp(target_log_probs)
        m_l = (target_probs * (u + (l == u).double() - b)).reshape(-1, num_bins, 1)
        m_u = (target_probs * (b - l)).reshape(-1, num_bins, 1)
        target_probs = torch.sum(m_l * l_mask + m_u * u_mask, axis=1)

    loss = -torch.mean(torch.sum(target_probs * pred_log_probs, axis=1))

    return loss, {"target_probs": target_probs}


class SIMBAV2(Agent):
    def __init__(
        self,
        models: Mapping[str, Model],
        memory: Optional[Union[Memory, Tuple[Memory]]] = None,
        observation_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        action_space: Optional[Union[int, Tuple[int], gymnasium.Space]] = None,
        device: Optional[Union[str, torch.device]] = None,
        cfg: Optional[dict] = None,
    ) -> None:
        _cfg = copy.deepcopy(SIMBAV2_DEFAULT_CONFIG)
        _cfg.update(cfg if cfg is not None else {})
        super().__init__(
            models=models,
            memory=memory,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            cfg=_cfg,
        )

        # models
        self.policy: GaussianMixin = self.models.get("policy", None)
        self.critic_1: DeterministicMixin = self.models.get("critic_1", None)
        self.critic_2: DeterministicMixin = self.models.get("critic_2", None)
        self.target_critic_1: DeterministicMixin = self.models.get("target_critic_1", None)
        self.target_critic_2: DeterministicMixin = self.models.get("target_critic_2", None)

        # checkpoint models
        self.checkpoint_modules["policy"] = self.policy
        self.checkpoint_modules["critic_1"] = self.critic_1
        self.checkpoint_modules["critic_2"] = self.critic_2
        self.checkpoint_modules["target_critic_1"] = self.target_critic_1
        self.checkpoint_modules["target_critic_2"] = self.target_critic_2

        # broadcast models' parameters in distributed runs
        if config.torch.is_distributed:
            logger.info(f"Broadcasting models' parameters")
            if self.policy is not None:
                self.policy.broadcast_parameters()
            if self.critic_1 is not None:
                self.critic_1.broadcast_parameters()
            if self.critic_2 is not None:
                self.critic_2.broadcast_parameters()

        if self.target_critic_1 is not None and self.target_critic_2 is not None:
            # freeze target networks with respect to optimizers (update via .update_parameters())
            self.target_critic_1.freeze_parameters(True)
            self.target_critic_2.freeze_parameters(True)

            # update target networks (hard update)
            self.target_critic_1.update_parameters(self.critic_1, polyak=1)
            self.target_critic_2.update_parameters(self.critic_2, polyak=1)

        # configuration
        self._gradient_steps = self.cfg["gradient_steps"]
        self._batch_size = self.cfg["batch_size"]

        self._discount_factor = self.cfg["discount_factor"]
        self._polyak = self.cfg["polyak"]

        self._actor_learning_rate = self.cfg["actor_learning_rate"]
        self._critic_learning_rate = self.cfg["critic_learning_rate"]
        self._learning_rate_scheduler = self.cfg["learning_rate_scheduler"]

        self._state_preprocessor = self.cfg["state_preprocessor"]

        self._random_timesteps = self.cfg["random_timesteps"]
        self._learning_starts = self.cfg["learning_starts"]

        self._grad_norm_clip = self.cfg["grad_norm_clip"]

        self._entropy_learning_rate = self.cfg["entropy_learning_rate"]
        self._learn_entropy = self.cfg["learn_entropy"]
        self._entropy_coefficient = self.cfg["initial_entropy_value"]

        self._normalize_weights = self.cfg["normalize_weights"]
        self._min_v = self.cfg["min_v"]
        self._max_v = self.cfg["max_v"]
        self._num_bins = self.cfg["num_bins"]
        self._use_categorical_critic = self.cfg["use_categorical_critic"]

        self.bin_values = torch.linspace(
            self._min_v, self._max_v, self._num_bins, device=self.device
        ).reshape(1, -1)

        self._rewards_shaper = self.cfg["rewards_shaper"]

        self._mixed_precision = self.cfg["mixed_precision"]

        # set up automatic mixed precision
        self._device_type = torch.device(device).type
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type, enabled=self._mixed_precision
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(enabled=self._mixed_precision)

        # entropy
        if self._learn_entropy:
            self._target_entropy = self.cfg["target_entropy"]
            if self._target_entropy is None:
                if issubclass(type(self.action_space), gymnasium.spaces.Box):
                    self._target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                elif issubclass(type(self.action_space), gymnasium.spaces.Discrete):
                    self._target_entropy = -self.action_space.n
                else:
                    self._target_entropy = 0

            self.log_entropy_coefficient = torch.log(
                torch.ones(1, device=self.device) * self._entropy_coefficient
            ).requires_grad_(True)
            self.entropy_optimizer = torch.optim.Adam(
                [self.log_entropy_coefficient], lr=self._entropy_learning_rate
            )

            self.checkpoint_modules["entropy_optimizer"] = self.entropy_optimizer

        # set up optimizers and learning rate schedulers
        if self.policy is not None and self.critic_1 is not None and self.critic_2 is not None:
            self.policy_optimizer = torch.optim.Adam(
                self.policy.parameters(), lr=self._actor_learning_rate
            )
            self.critic_optimizer = torch.optim.Adam(
                itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                lr=self._critic_learning_rate,
            )
            if self._learning_rate_scheduler is not None:
                self.policy_scheduler = self._learning_rate_scheduler(
                    self.policy_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )
                self.critic_scheduler = self._learning_rate_scheduler(
                    self.critic_optimizer, **self.cfg["learning_rate_scheduler_kwargs"]
                )

            self.checkpoint_modules["policy_optimizer"] = self.policy_optimizer
            self.checkpoint_modules["critic_optimizer"] = self.critic_optimizer

        # set up preprocessors
        if self._state_preprocessor:
            self._state_preprocessor = self._state_preprocessor(
                **self.cfg["state_preprocessor_kwargs"]
            )
            self.checkpoint_modules["state_preprocessor"] = self._state_preprocessor
        else:
            self._state_preprocessor = self._empty_preprocessor

    def init(self, trainer_cfg: Optional[Mapping[str, Any]] = None) -> None:
        """Initialize the agent"""
        super().init(trainer_cfg=trainer_cfg)
        self.set_mode("eval")

        # create tensors in memory
        if self.memory is not None:
            self.memory.create_tensor(
                name="states", size=self.observation_space, dtype=torch.float32
            )
            self.memory.create_tensor(
                name="next_states", size=self.observation_space, dtype=torch.float32
            )
            self.memory.create_tensor(name="actions", size=self.action_space, dtype=torch.float32)
            self.memory.create_tensor(name="rewards", size=1, dtype=torch.float32)
            self.memory.create_tensor(name="terminated", size=1, dtype=torch.bool)
            self.memory.create_tensor(name="truncated", size=1, dtype=torch.bool)

            self._tensors_names = [
                "states",
                "actions",
                "rewards",
                "next_states",
                "terminated",
                "truncated",
            ]

    def act(self, states: torch.Tensor, timestep: int, timesteps: int) -> torch.Tensor:
        """Process the environment's states to make a decision (actions) using the main policy

        :param states: Environment's states
        :type states: torch.Tensor
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int

        :return: Actions
        :rtype: torch.Tensor
        """
        # sample random actions
        # TODO, check for stochasticity
        if timestep < self._random_timesteps:
            return self.policy.random_act(
                {"states": self._state_preprocessor(states)}, role="policy"
            )

        # sample stochastic actions
        with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
            actions, log_probs, outputs = self.policy.act(
                {"states": self._state_preprocessor(states)}, role="policy"
            )

        return actions, log_probs, outputs

    def record_transition(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: Any,
        timestep: int,
        timesteps: int,
    ) -> None:
        """Record an environment transition in memory

        :param states: Observations/states of the environment used to make the decision
        :type states: torch.Tensor
        :param actions: Actions taken by the agent
        :type actions: torch.Tensor
        :param rewards: Instant rewards achieved by the current actions
        :type rewards: torch.Tensor
        :param next_states: Next observations/states of the environment
        :type next_states: torch.Tensor
        :param terminated: Signals to indicate that episodes have terminated
        :type terminated: torch.Tensor
        :param truncated: Signals to indicate that episodes have been truncated
        :type truncated: torch.Tensor
        :param infos: Additional information about the environment
        :type infos: Any type supported by the environment
        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        super().record_transition(
            states, actions, rewards, next_states, terminated, truncated, infos, timestep, timesteps
        )

        if self.memory is not None:
            # reward shaping
            if self._rewards_shaper is not None:
                rewards = self._rewards_shaper(rewards, timestep, timesteps)

            # storage transition in memory
            self.memory.add_samples(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
            )
            for memory in self.secondary_memories:
                memory.add_samples(
                    states=states,
                    actions=actions,
                    rewards=rewards,
                    next_states=next_states,
                    terminated=terminated,
                    truncated=truncated,
                )

    def pre_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called before the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        pass

    def post_interaction(self, timestep: int, timesteps: int) -> None:
        """Callback called after the interaction with the environment

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """
        if timestep >= self._learning_starts:
            self.set_mode("train")
            self._update(timestep, timesteps)
            self.set_mode("eval")

        # write tracking data and checkpoints
        super().post_interaction(timestep, timesteps)

    def _process_categorical_critic_outputs(
        self,
        outputs: torch.Tensor,
    ) -> torch.Tensor:
        log_prob = F.log_softmax(outputs, dim=1)

        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=1)

        return value, log_prob

    def _update(self, timestep: int, timesteps: int) -> None:
        """Algorithm's main update step

        :param timestep: Current timestep
        :type timestep: int
        :param timesteps: Number of timesteps
        :type timesteps: int
        """

        # gradient steps
        for gradient_step in range(self._gradient_steps):

            # sample a batch from memory
            (
                sampled_states,
                sampled_actions,
                sampled_rewards,
                sampled_next_states,
                sampled_terminated,
                sampled_truncated,
            ) = self.memory.sample(names=self._tensors_names, batch_size=self._batch_size)[0]

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                sampled_states = self._state_preprocessor(sampled_states, train=True)
                sampled_next_states = self._state_preprocessor(sampled_next_states, train=True)

                if self._use_categorical_critic:
                    with torch.no_grad():
                        next_actions, next_log_prob, _ = self.policy.act(
                            {"states": sampled_next_states}, role="policy"
                        )

                        target_critic_1_out, _, _ = self.target_critic_1.act(
                            {"states": sampled_next_states, "taken_actions": next_actions},
                            role="target_critic_1",
                        )
                        target_q1, target_log_prob_q1 = self._process_categorical_critic_outputs(
                            target_critic_1_out
                        )
                        target_critic_2_out, _, _ = self.target_critic_2.act(
                            {"states": sampled_next_states, "taken_actions": next_actions},
                            role="target_critic_2",
                        )
                        target_q2, target_log_prob_q2 = self._process_categorical_critic_outputs(
                            target_critic_2_out
                        )

                        target_log_prob = torch.where(
                            (target_q1 < target_q2).unsqueeze(-1),
                            target_log_prob_q1,
                            target_log_prob_q2,
                        )

                    critic_1_out, _, _ = self.critic_1.act(
                        {"states": sampled_states, "taken_actions": sampled_actions},
                        role="critic_1",
                    )
                    _, log_prob_q1 = self._process_categorical_critic_outputs(critic_1_out)
                    critic_2_out, _, _ = self.critic_2.act(
                        {"states": sampled_states, "taken_actions": sampled_actions},
                        role="critic_2",
                    )
                    _, log_prob_q2 = self._process_categorical_critic_outputs(critic_2_out)
                    loss_1, info_1 = categorical_td_loss(
                        log_prob_q1,
                        target_log_prob,
                        sampled_rewards,
                        (sampled_terminated | sampled_truncated).double(),
                        next_log_prob,
                        self._entropy_coefficient,
                        self._discount_factor,
                        self._num_bins,
                        self._min_v,
                        self._max_v,
                        self.device,
                    )
                    loss_2, info_2 = categorical_td_loss(
                        log_prob_q2,
                        target_log_prob,
                        sampled_rewards,
                        (sampled_terminated | sampled_truncated).double(),
                        next_log_prob,
                        self._entropy_coefficient,
                        self._discount_factor,
                        self._num_bins,
                        self._min_v,
                        self._max_v,
                        self.device,
                    )

                    target_values = torch.sum(info_1["target_probs"] * self.bin_values, dim=1)

                    critic_loss = torch.mean(loss_1 + loss_2)
                else:
                    # compute target values
                    with torch.no_grad():
                        next_actions, next_log_prob, _ = self.policy.act(
                            {"states": sampled_next_states}, role="policy"
                        )

                        target_q1_values, _, _ = self.target_critic_1.act(
                            {"states": sampled_next_states, "taken_actions": next_actions},
                            role="target_critic_1",
                        )
                        target_q2_values, _, _ = self.target_critic_2.act(
                            {"states": sampled_next_states, "taken_actions": next_actions},
                            role="target_critic_2",
                        )

                        target_q_values = (
                            torch.min(target_q1_values, target_q2_values)
                            - self._entropy_coefficient * next_log_prob
                        )
                        target_values = (
                            sampled_rewards
                            + self._discount_factor
                            * (sampled_terminated | sampled_truncated).logical_not()
                            * target_q_values
                        )

                    # compute critic loss
                    critic_1_values, _, _ = self.critic_1.act(
                        {"states": sampled_states, "taken_actions": sampled_actions},
                        role="critic_1",
                    )
                    critic_2_values, _, _ = self.critic_2.act(
                        {"states": sampled_states, "taken_actions": sampled_actions},
                        role="critic_2",
                    )

                    critic_loss = (
                        F.mse_loss(critic_1_values, target_values)
                        + F.mse_loss(critic_2_values, target_values)
                    ) / 2

            # optimization step (critic)
            self.critic_optimizer.zero_grad()
            self.scaler.scale(critic_loss).backward()

            if config.torch.is_distributed:
                self.critic_1.reduce_parameters()
                self.critic_2.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.critic_optimizer)
                nn.utils.clip_grad_norm_(
                    itertools.chain(self.critic_1.parameters(), self.critic_2.parameters()),
                    self._grad_norm_clip,
                )

            self.scaler.step(self.critic_optimizer)
            if self._normalize_weights:
                l2normalize_model(self.critic_1)
                l2normalize_model(self.critic_2)

            with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                # compute policy (actor) loss
                actions, log_prob, _ = self.policy.act({"states": sampled_states}, role="policy")
                critic_1_values, _, _ = self.critic_1.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic_1"
                )
                critic_2_values, _, _ = self.critic_2.act(
                    {"states": sampled_states, "taken_actions": actions}, role="critic_2"
                )
                if self._use_categorical_critic:
                    critic_1_values, _ = self._process_categorical_critic_outputs(critic_1_values)
                    critic_2_values, _ = self._process_categorical_critic_outputs(critic_2_values)

                policy_loss = (
                    self._entropy_coefficient * log_prob
                    - torch.min(critic_1_values, critic_2_values)
                ).mean()

            # optimization step (policy)
            self.policy_optimizer.zero_grad()
            self.scaler.scale(policy_loss).backward()

            if config.torch.is_distributed:
                self.policy.reduce_parameters()

            if self._grad_norm_clip > 0:
                self.scaler.unscale_(self.policy_optimizer)
                nn.utils.clip_grad_norm_(self.policy.parameters(), self._grad_norm_clip)

            self.scaler.step(self.policy_optimizer)
            if self._normalize_weights:
                l2normalize_model(self.policy)

            # entropy learning
            if self._learn_entropy:
                with torch.autocast(device_type=self._device_type, enabled=self._mixed_precision):
                    # compute entropy loss
                    entropy_loss = -(
                        self.log_entropy_coefficient * (log_prob + self._target_entropy).detach()
                    ).mean()

                # optimization step (entropy)
                self.entropy_optimizer.zero_grad()
                self.scaler.scale(entropy_loss).backward()
                self.scaler.step(self.entropy_optimizer)

                # compute entropy coefficient
                self._entropy_coefficient = torch.exp(self.log_entropy_coefficient.detach())

            policy_entropy = -log_prob
            policy_entropy_scaled = -(self._entropy_coefficient * log_prob)

            self.scaler.update()  # called once, after optimizers have been stepped

            # update target networks
            self.target_critic_1.update_parameters(self.critic_1, polyak=self._polyak)
            self.target_critic_2.update_parameters(self.critic_2, polyak=self._polyak)

            # update learning rate
            if self._learning_rate_scheduler:
                self.policy_scheduler.step()
                self.critic_scheduler.step()

            # record data
            if self.write_interval > 0:
                self.track_data("Loss / Policy loss", policy_loss.item())
                self.track_data("Loss / Critic loss", critic_loss.item())

                self.track_data("Q-network / Q1 (max)", torch.max(critic_1_values).item())
                self.track_data("Q-network / Q1 (min)", torch.min(critic_1_values).item())
                self.track_data("Q-network / Q1 (mean)", torch.mean(critic_1_values).item())

                self.track_data("Q-network / Q2 (max)", torch.max(critic_2_values).item())
                self.track_data("Q-network / Q2 (min)", torch.min(critic_2_values).item())
                self.track_data("Q-network / Q2 (mean)", torch.mean(critic_2_values).item())

                self.track_data("Target / Target (max)", torch.max(target_values).item())
                self.track_data("Target / Target (min)", torch.min(target_values).item())
                self.track_data("Target / Target (mean)", torch.mean(target_values).item())

                self.track_data("Policy / Entropy (max)", torch.max(policy_entropy).item())
                self.track_data("Policy / Entropy (min)", torch.min(policy_entropy).item())
                self.track_data("Policy / Entropy (mean)", torch.mean(policy_entropy).item())

                self.track_data(
                    "Policy / Entropy Scaled (max)", torch.max(policy_entropy_scaled).item()
                )
                self.track_data(
                    "Policy / Entropy Scaled (min)", torch.min(policy_entropy_scaled).item()
                )
                self.track_data(
                    "Policy / Entropy Scaled (mean)", torch.mean(policy_entropy_scaled).item()
                )

                if self._learn_entropy:
                    self.track_data("Loss / Entropy loss", entropy_loss.item())
                    self.track_data(
                        "Coefficient / Entropy coefficient", self._entropy_coefficient.item()
                    )

                if self._learning_rate_scheduler:
                    self.track_data(
                        "Learning / Policy learning rate", self.policy_scheduler.get_last_lr()[0]
                    )
                    self.track_data(
                        "Learning / Critic learning rate", self.critic_scheduler.get_last_lr()[0]
                    )

                for i in range(self.action_space.shape[0]):
                    self.track_histogram_data(f"Policy / Action Distribution {i}", next_actions[:, i])
