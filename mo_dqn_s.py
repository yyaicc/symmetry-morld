"""EUPG is an ESR algorithm based on Policy Gradient (REINFORCE like)."""
import time
import random
from copy import deepcopy
from typing import Callable, List, Optional, Union
from typing_extensions import override

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import wandb
from collections import deque
from torch.distributions import Categorical

from symmetrizer.nn import BasisFourRoomNetworkWrapper, BasisFourRoomLayer, BasisDstNetworkWrapper, BasisDstLayer,  \
    BasisLinear, BasisReacherNetworkWrapper, BasisReacherLayer, BasisLunarLanderNetworkWrapper, BasisLunarLanderLayer


from morl_baselines.common.accrued_reward_buffer import AccruedRewardReplayBuffer
from morl_baselines.common.evaluation import log_episode_info
from morl_baselines.common.morl_algorithm import MOAgent, MOPolicy
from morl_baselines.common.networks import layer_init, mlp


class n_step_replay_buffer(object):
    def __init__(self, buffer_size, n_step, gamma):
        self.buffer_size = buffer_size
        self.n_step = n_step
        self.gamma = gamma
        self.memory = deque(maxlen=self.buffer_size)
        self.n_step_buffer = deque(maxlen=self.n_step)

    def _get_n_step_info(self):
        reward, next_observation, done = self.n_step_buffer[-1][-3:]
        for _, _, rew, next_obs, do in reversed(list(self.n_step_buffer)[: -1]):
            reward = self.gamma * reward * (1 - do) + rew
            next_observation, done = (next_obs, do) if do else (next_observation, done)
        return reward, next_observation, done

    def add(self, observation, action, reward, next_observation, done):
        observation = np.expand_dims(observation, 0)
        next_observation = np.expand_dims(next_observation, 0)

        self.n_step_buffer.append([observation, action, reward, next_observation, done])
        if len(self.n_step_buffer) < self.n_step:
            return
        reward, next_observation, done = self._get_n_step_info()
        observation, action = self.n_step_buffer[0][: 2]
        self.memory.append([observation, action, reward, next_observation, done])

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        observation, action, reward, next_observation, done = zip(* batch)
        return np.concatenate(observation, 0), action, reward, np.concatenate(next_observation, 0), done

    def __len__(self):
        return len(self.memory)

class DqnNet(nn.Module):
    """Policy network."""

    def __init__(self, obs_shape, action_dim, net_arch):
        """Initialize the policy network.

        Args:
            obs_shape: Observation shape
            action_dim: Action dimension
            net_arch: Number of units per layer
        """
        super().__init__()
        self.obs_shape = obs_shape[0] if len(obs_shape) == 1 else obs_shape[0]*obs_shape[1]
        self.action_dim = action_dim

        if len(obs_shape) == 1:
            input_dim = obs_shape[0]
        elif len(obs_shape) > 1:
            input_dim = obs_shape[0]*obs_shape[1]

        # |S|+|R| -> ... -> |A|
        self.net = mlp(input_dim, action_dim, net_arch, activation_fn=nn.Tanh)
        self.apply(layer_init)

    def forward(self, obs: th.Tensor):

        obs = obs.reshape(-1, self.obs_shape)
        x = self.net(obs)

        return x  # Batch Size x |Actions|

    def act(self, obs: th.Tensor, epsilon: float=1.0):
        if random.random() > epsilon:
            q_value = self.forward(obs).squeeze().view(-1, self.action_dim)
            action = q_value.max(1)[1].data[0].item()
            #distribution = Categorical(probas)
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

class DqnNet_S(nn.Module):

    """Actor-Critic network."""

    def __init__(
        self,
        obs_shape: tuple,
        action_dim: tuple,
        net_arch: list = [],
        env_name: str = "env_name",
    ):
        """Initialize the network.

        Args:
            obs_shape: Observation shape
            action_dim: Action shape
            hidden_dim: Number of units per layer
        """
        super().__init__()

        self.obs_shape = obs_shape[0] if len(obs_shape) == 1 else obs_shape[0] * obs_shape[1]
        self.action_dim = action_dim

        input_size = 1
        fc_sizes = net_arch
        basis = "equivariant"
        gain_type = "xavier"

        if "deep" in env_name:
            self.head = BasisDstNetworkWrapper(input_size, fc_sizes,
                                                 gain_type=gain_type,
                                                 basis=basis)
            self.pi = BasisDstLayer(fc_sizes[-1], 1,
                                             gain_type=gain_type,
                                             basis=basis)

        elif env_name == "mo-lunar-lander-v2":
            self.head = BasisLunarLanderNetworkWrapper(input_size, fc_sizes,
                                                 gain_type=gain_type,
                                                 basis=basis)
            self.pi = BasisLunarLanderLayer(fc_sizes[-1], 1,
                                             gain_type=gain_type,
                                             basis=basis)

        elif env_name == "mo-reacher-v4":
            self.head = BasisReacherNetworkWrapper(input_size, fc_sizes,
                                                 gain_type=gain_type,
                                                 basis=basis)
            self.pi = BasisReacherLayer(fc_sizes[-1], 1,
                                             gain_type=gain_type,
                                             basis=basis)
            # self.value = BasisReacherLayer(fc_sizes[-1], 1,
            #                                     gain_type=gain_type,
            #                                     basis=basis, out="invariant")

        elif env_name == "four-room-v0":
            self.head = BasisFourRoomNetworkWrapper(input_size, fc_sizes,
                                                   gain_type=gain_type,
                                                   basis=basis)
            self.pi = BasisFourRoomLayer(fc_sizes[-1], 1,
                                        gain_type=gain_type,
                                        basis=basis)
            # self.value = BasisReacherLayer(fc_sizes[-1], 1,
            #                                     gain_type=gain_type,
            #                                     basis=basis, out="invariant")

    def forward(self, obs: th.Tensor):
        """Get the action and value of an observation.

        Args:
            obs: Observation
            action: Action. If None, a new action is sampled.

        Returns: A tuple of (action, logprob, entropy, value)
        """

        obs = obs.reshape(-1, self.obs_shape)
        base = self.head(obs)
        x = self.pi(base)

        #action_probs = F.softmax(self.head(obs), dim=-1).squeeze()

        return x

    def act(self, obs: th.Tensor, epsilon: float=1.0):
        if random.random() > epsilon:
            q_value = self.forward(obs).squeeze().view(-1, self.action_dim)
            action = q_value.max(1)[1].data[0].item()
            #distribution = Categorical(probas)
        else:
            action = random.choice(list(range(self.action_dim)))
        return action

class DQN_S(MOPolicy, MOAgent):

    """Expected Utility Policy Gradient Algorithm with Baseline.

    The idea is to condition the network on the accrued reward and to scalarize the rewards based on the episodic return (accrued + future rewards)
    Paper: D. Roijers, D. Steckelmacher, and A. Nowe, Multi-objective Reinforcement Learning for the Expected Utility of the Return. 2018.
    """

    def __init__(
        self,
        env: gym.Env,
        scalarization: Callable[[np.ndarray, np.ndarray], float],
        #env_name: str = "CartPole-v0",
        symmetry: bool=True,
        weights: np.ndarray = np.ones(2),
        id: Optional[int] = None,
        exploration=200,
        epsilon_init = 1.0,
        epsilon_min = 0.01,
        decay = 0.99,
        n_step = 2,
        soft_update_freq = 30,
        batch_size = 64,
        buffer_size: int = int(1e7),
        net_arch: List = [64, 64],
        gamma: float = 0.99,
        learning_rate: float = 1e-4,
        project_name: str = "MORL-DQN",
        experiment_name: str = "DQN",
        wandb_entity: Optional[str] = None,
        log: bool = True,
        log_every: int = 1000,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
        parent_rng: Optional[np.random.Generator] = None,
    ):
        """Initialize the EUPG algorithm.

        Args:
            env: Environment
            scalarization: Scalarization function to use (can be non-linear)
            weights: Weights to use for the scalarization function
            id: Id of the agent (for logging)
            buffer_size: Size of the replay buffer
            net_arch: Number of units per layer
            gamma: Discount factor
            learning_rate: Learning rate (alpha)
            project_name: Name of the project (for logging)
            experiment_name: Name of the experiment (for logging)
            wandb_entity: Entity to use for wandb
            log: Whether to log or not
            log_every: Log every n episodes
            device: Device to use for NN. Can be "cpu", "cuda" or "auto".
            seed: Seed for the random number generator
            parent_rng: Parent random number generator (for reproducibility)
        """
        MOAgent.__init__(self, env, device, seed=seed)
        MOPolicy.__init__(self, None, device)

        # Seeding
        self.seed = seed
        self.parent_rng = parent_rng
        if parent_rng is not None:
            self.np_random = parent_rng
        else:
            self.np_random = np.random.default_rng(self.seed)

        self.env = env
        self.id = id
        self.num_episodes = 0
        self.symmetry = symmetry


        #para
        self.exploration=exploration
        self.epsilon_init = epsilon_init
        self.epsilon_min = epsilon_min
        self.decay = decay
        self.n_step = n_step
        self.soft_update_freq = soft_update_freq
        self.batch_size = batch_size


        # RL
        self.scalarization = scalarization
        self.weights = weights
        self.gamma = gamma

        # Learning
        self.buffer_size = buffer_size
        self.net_arch = net_arch
        self.learning_rate = learning_rate

        self.buffer = n_step_replay_buffer(self.buffer_size, n_step, gamma)
        self.loss_fn = nn.MSELoss()

        if self.symmetry:
            self.net = DqnNet_S(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                env_name=experiment_name,
                net_arch=net_arch
            ).to(self.device)

            self.target_net = DqnNet_S(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                env_name=experiment_name,
                net_arch=net_arch
            ).to(self.device)
        else:
            self.net = DqnNet(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                net_arch=net_arch
            ).to(self.device)

            self.target_net = DqnNet(
                obs_shape=self.observation_shape,
                action_dim=self.action_dim,
                net_arch=net_arch
            ).to(self.device)

        #print("id", id)
        # self.net = PolicyNet(
        #     obs_shape=self.observation_shape,
        #     rew_dim=self.reward_dim,
        #     action_dim=self.action_dim,
        #     net_arch=self.net_arch,
        # ).to(self.device)

        # self.baseline = ValueNet(
        #     obs_shape=self.observation_shape,
        #     rew_dim=self.reward_dim,
        #     net_arch = self.net_arch
        # ).to(self.device)

        self.optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)
        #self.v_optimizer = optim.Adam(self.baseline.parameters(), lr=2e-4)

        # Logging
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.log = log
        self.log_every = log_every
        if log and parent_rng is None:
            self.setup_wandb(self.project_name, self.experiment_name, wandb_entity)

        self.init_train()

    def __deepcopy__(self, memo):
        """Deep copy the policy."""
        copied_net = deepcopy(self.net)
        copied = type(self)(
            self.env,
            self.scalarization,
            #self.env_name,
            self.symmetry,
            self.weights,
            self.id,
            self.exploration,
            self.epsilon_init,
            self.epsilon_min,
            self.soft_update_freq,
            self.batch_size,
            self.decay,
            self.n_step,
            self.buffer_size,
            self.net_arch,
            self.gamma,
            self.learning_rate,
            self.project_name,
            self.experiment_name,
            log=self.log,
            device=self.device,
            parent_rng=self.parent_rng,
        )

        copied.global_step = self.global_step
        copied.optimizer = optim.Adam(copied_net.parameters(), lr=self.learning_rate)
        copied.buffer = deepcopy(self.buffer)
        #copied.target_net = deepcopy(self.target_net)
        return copied

    @override
    def get_policy_net(self) -> nn.Module:
        return self.net

    @override
    def get_buffer(self):
        return self.buffer

    @override
    def set_buffer(self, buffer):
        raise Exception("On-policy algorithms should not share buffer.")

    @override
    def set_weights(self, weights: np.ndarray):
        self.weights = weights

    @th.no_grad()
    @override
    def eval(self, obs: np.ndarray, accrued_reward: Optional[np.ndarray]) -> Union[int, np.ndarray]:
        if type(obs) is int:
            obs = th.as_tensor([obs]).float().to(self.device)
        else:
            obs = th.as_tensor(obs).float().to(self.device)

        q_value  = self.net(obs).squeeze().view(-1, self.action_dim)
        action = q_value.max(1)[1].data[0].item()
        #distribution = Categorical(probas)
        return action

    @th.no_grad()
    def __choose_action(self, obs: th.Tensor, accrued_reward: th.Tensor) -> int:

        action = self.net.distribution(obs, accrued_reward)
        action = action.sample().detach().item()
        #print(action)
        return action

    @override
    def update(self):

        observation, action, reward, next_observation, done = self.buffer.sample(self.batch_size)

        observation = th.FloatTensor(observation).to(self.device)
        action = th.LongTensor(action).to(self.device)
        reward = th.FloatTensor(reward).to(self.device)
        next_observation = th.FloatTensor(next_observation).to(self.device)
        done = th.FloatTensor(done).to(self.device)

        q_values = self.net.forward(observation).squeeze().view(-1, self.action_dim)
        next_q_values = self.target_net.forward(next_observation).squeeze().view(-1, self.action_dim)
        argmax_actions = self.net.forward(next_observation).squeeze().view(-1, self.action_dim).max(1)[1].detach()

        next_q_value = next_q_values.gather(1, argmax_actions.unsqueeze(1)).squeeze(1)
        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        expected_q_value = reward + (self.gamma ** self.n_step) * (1 - done) * next_q_value

        # loss = loss_fn(q_value, expected_q_value.detach())
        loss = (expected_q_value.detach() - q_value).pow(2)
        loss = loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.num_episodes % self.soft_update_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())

    def init_train(self):

        for i in range(self.exploration):
            # Init
            (
                obs,
                _,
            ) = self.env.reset()

            with th.no_grad():
                # For training, takes action according to the policy
                action = self.net.act(th.Tensor(obs).to(self.device))
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            sca_reward = vec_reward @ self.weights
            # Memory update
            self.buffer.add(obs, action, sca_reward, next_obs, terminated)

    def train(self, total_timesteps: int, eval_env: Optional[gym.Env] = None, eval_freq: int = 1000, start_time=None):
        """Train the agent.

        Args:
            total_timesteps: Number of timesteps to train for
            eval_env: Environment to run policy evaluation on
            eval_freq: Frequency of policy evaluation
            start_time: Start time of the training (for SPS)
        """
        if start_time is None:
            start_time = time.time()
        # Init
        (
            obs,
            _,
        ) = self.env.reset()

        epsilon = self.epsilon_init
        # Training loop
        for _ in range(1, total_timesteps + 1):
        #while not (terminated or truncated):
            self.global_step += 1

            with th.no_grad():
                # For training, takes action according to the policy
                action = self.net.act(th.Tensor(obs).to(self.device), epsilon)
            next_obs, vec_reward, terminated, truncated, info = self.env.step(action)

            sca_reward = vec_reward @ self.weights
            # Memory update
            self.buffer.add(obs, action, sca_reward, next_obs, terminated)

            self.update()

            if terminated or truncated:

                if epsilon > self.epsilon_min:
                    epsilon = epsilon * self.decay

                obs, _ = self.env.reset()
                self.num_episodes += 1

            else:
                obs = next_obs

            if self.log and self.global_step % 1000 == 0:
                print("SPS:", int(self.global_step / (time.time() - start_time)))
                wandb.log({"charts/SPS": int(self.global_step / (time.time() - start_time)), "global_step": self.global_step})

    @override
    def get_config(self) -> dict:
        return {
            "env_id": self.env.unwrapped.spec.id,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "gamma": self.gamma,
            "net_arch": self.net_arch,
            "seed": self.seed,
        }
