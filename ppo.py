import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# 超参数
learning_rate = 0.0005
gamma = 0.98
lmbda = 0.95
eps_clip = 0.1
K_epoch = 3
T_horizon = 20  # 表示经验收集的时间步长度（Trajectory Horizon），即每次与环境交互多少步后，才做一次策略更新


class PPO(nn.Module):
    def __init__(self):
        super(PPO, self).__init__()
        self.data = []

        self.fc_base = nn.Linear(4, 256)
        self.fc_actor = nn.Linear(256, 2)
        self.fc_critic = nn.Linear(256, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    #  徒弟（Actor）决定怎么摊煎饼（输出动作概率）
    def actor(self, x, softmax_dim=0):
        x = F.relu(self.fc_base(x))  # 观察面粉状态（输入状态x）
        x = self.fc_actor(x)  # 想尝试的摊法（输出原始动作logits）
        # 在 softmax_dim = 0 维度计算
        # F.softmax(x, dim=0) 计算 Softmax，输出 [0.6, 0.4]，所有值加起来等于 1。
        prob = F.softmax(x, dim=softmax_dim)  # 计算摊薄/摊厚的概率（如[0.7, 0.3]）
        return prob

    # 师父（Critic）	评价当前摊法能得多少分（状态价值）
    def critic(self, x):
        x = F.relu(self.fc_base(x))  # 同样观察面粉状态
        v = self.fc_critic(x)  # 预测这锅能卖多少钱（状态价值）
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_list, a_list, r_list, s_prime_list, prob_a_list, done_list = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_list.append(s)
            a_list.append([a])
            r_list.append([r])
            s_prime_list.append(s_prime)
            prob_a_list.append([prob_a])
            done_mask = 0 if done else 1
            done_list.append([done_mask])

        s, a, r, s_prime, done_mask, prob_a = torch.tensor(s_list, dtype=torch.float), torch.tensor(a_list), \
            torch.tensor(r_list), torch.tensor(s_prime_list, dtype=torch.float), \
            torch.tensor(done_list, dtype=torch.float), torch.tensor(prob_a_list)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()
        for i in range(K_epoch):
            """
                通过优势函数 advantage = td_target - v(s) 告诉徒弟：
                如果 advantage > 0：这次摊得比预期好，继续保持！
                如果 advantage < 0：这次摊得不行，下次少这么干！
            """
            td_target = r + gamma * self.critic(s_prime) * done_mask  # 这锅实际赚的钱 + 下锅预期
            delta = td_target - self.critic(s)
            delta = delta.detach().numpy()

            # GAE优势函数，考虑连续多锅的惊喜值（不只是当前锅）
            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            actor = self.actor(s, softmax_dim=1)
            actor_a = actor.gather(1, a)
            ratio = torch.exp(torch.log(actor_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage  # 不许超过±10%的改动
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.critic(s), td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()


def main():
    env = gym.make('CartPole-v1', render_mode="human")
    model = PPO()
    score = 0.0
    print_interval = 20

    for n_epi in range(10000):
        s, _ = env.reset()
        env.render()
        done = False
        while not done:
            for t in range(T_horizon):
                # 生成归一化的概率，每行和为 1
                prob = model.actor(torch.from_numpy(s).float())
                # prob 对应 Categorical 对象表示一个有 m 个可能类别，且必须满足（必须满足 sum(probs)=1）
                # 造骰子，并设定其概率
                m = Categorical(prob)
                # 调用 .sample() 时，它会按概率随机选取一个类别
                # .item() 是 PyTorch 张量（Tensor） 的一个方法，它用于将 单个数值张量 转换为 Python 标量（int 或 float）
                # m.sample()表示扔一次骰子
                a = m.sample().item()
                # a表示动作(要么是 0 ,要么是 1 )
                # s_prime表示当前状态
                s_prime, r, done, truncated, info = env.step(a)
                # 上一时刻状态，上一时刻动作，回报，当前的状态，预测的概率(暂时还不清楚)
                model.put_data((s, a, r / 100.0, s_prime, prob[a].item(), done))
                s = s_prime

                score += r
                if done:
                    break

            model.train_net()

        if n_epi % print_interval == 0 and n_epi != 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score / print_interval))
            score = 0.0

    env.close()


if __name__ == '__main__':
    main()
