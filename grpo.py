import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical

# The key differences in the hyperparameters for GRPO are the KL target and KL beta
# The KL target is the target KL divergence for adaptive adjustment
# The KL beta is the initial weight of the KL penalty term

HYPERPARAMETERS_GRPO = {
    'LR': 2e-4,  # Learning rate
    'EPS_CLIP': 0.2,  # Clipping range (PPO) -- replaced in GRPO by KL regularization
    'GAMMA': 0.99,  # Discount factor
    'LAMBDA': 0.95,  # GAE parameter
    'K_EPOCHS': 20,  # Update epochs per optimization step
    'KL_TARGET': 0.05,  # Target KL divergence for adaptive adjustment
    'KL_BETA': 0.05,  # Initial weight of KL penalty term
    'LR_DECAY': 0.9,  # Decay factor for learning rate if KL exceeds threshold
    'T_MAX': 2000  # Maximum training steps per episode
}


class ActorCriticGRPO(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticGRPO, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        value = self.critic(state)
        return action_probs, value

class GRPO:
    def __init__(self, state_dim, action_dim):
        """
        Initialize the GRPO agent with the Actor-Critic network, optimizer, memory, and hyperparameters.
        """
        self.policy = ActorCriticGRPO(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=HYPERPARAMETERS_GRPO['LR'])
        self.memory = []
        self.gamma = HYPERPARAMETERS_GRPO['GAMMA']
        self.lam = HYPERPARAMETERS_GRPO['LAMBDA']
        self.kl_beta = HYPERPARAMETERS_GRPO['KL_BETA']
        self.action_dim = action_dim


    def select_action(self, state):
        state = torch.tensor(np.array(state), dtype=torch.float32)
        action_probs, _ = self.policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        # ✅ FIXED: Return action_probs so it can be stored in memory for KL computation
        return action.item(), log_prob, action_probs.detach()

    def store_transition(self, transition):
        self.memory.append(transition)

    def optimize(self):
        """
        Update the policy using stored memory transitions.
        """
        states, actions, log_probs_old, rewards, dones, old_action_probs = zip(*self.memory)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.int64)
        log_probs_old = torch.stack(log_probs_old).detach()
        old_action_probs = torch.stack(old_action_probs).detach()
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        # GAE computation
        advantages = []
        returns = []
        R = 0
        A = 0
        for r, d, v in zip(reversed(rewards), reversed(dones), reversed(old_action_probs)):
            R = r + self.gamma * R * (1 - d)
            returns.insert(0, R)
            delta = r + self.gamma * v.max() - R  # Ensure proper advantage computation
            A = delta + self.gamma * self.lam * A * (1 - d)
            advantages.insert(0, A)
        returns = torch.tensor(returns, dtype=torch.float32)
        advantages = torch.tensor(advantages, dtype=torch.float32)

        for _ in range(HYPERPARAMETERS_GRPO['K_EPOCHS']):
            action_probs, state_values = self.policy(states)
            dist = Categorical(action_probs)
            log_probs_new = dist.log_prob(actions)

            # KL divergence computation
            kl_divergence = torch.distributions.kl.kl_divergence(Categorical(old_action_probs), Categorical(action_probs))

            # ratio for policy loss
            ratio = torch.exp(log_probs_new - log_probs_old)

            # Policy loss with KL penalty
            policy_loss = -torch.mean(torch.min(ratio * advantages, self.kl_beta * kl_divergence))

            # Value loss
            value_loss = nn.MSELoss()(state_values.squeeze(), returns)

            # Total loss with KL penalty term
            loss = policy_loss + value_loss + self.kl_beta * kl_divergence.mean()

            # Adjust KL beta dynamically
            if kl_divergence.mean().item() > HYPERPARAMETERS_GRPO['KL_TARGET']:
                self.kl_beta *= 1.5  # Increase penalty
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] *= HYPERPARAMETERS_GRPO['LR_DECAY']  # Reduce learning rate
            elif kl_divergence.mean().item() < HYPERPARAMETERS_GRPO['KL_TARGET'] / 2:
                self.kl_beta /= 1.5  # Reduce penalty

            # Update parameters
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.memory.clear()


if __name__ == "__main__":
    # Run multiple seeds to analyze variance in rewards
    num_seeds = 5
    num_episodes = 200

    seeds = np.random.randint(0, 123456, num_seeds)

    num_episodes =100000 # Increase number of episodes for GRPO training

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)

        env = gym.make("CartPole-v1")
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        grpo_agent = GRPO(state_dim, action_dim)

        for episode in range(num_episodes):
            state = env.reset()

            if isinstance(state, tuple):
                state, _ = state  # Unpack (state, info) when gym returns a tuple
            state = np.array(state, dtype=np.float32)  # Ensure it's a proper NumPy array
            episode_reward = 0
            for t in range(HYPERPARAMETERS_GRPO['T_MAX']):
                action, log_prob, action_probs = grpo_agent.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated  # Ensure compatibility with new Gym versions
                grpo_agent.store_transition((state, action, log_prob, reward, done, action_probs))
                state = next_state
                episode_reward += reward
                if done:
                    break

            grpo_agent.optimize()
            if (episode + 1) % 10 == 0:
                print(f"Seed {seed} - Episode {episode+1}: Reward = {episode_reward}")
        env.close()

    print("Training complete.")