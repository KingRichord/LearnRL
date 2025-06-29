import gym
import collections
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 超参数
learning_rate = 0.0005
gamma         = 0.98
buffer_limit  = 50000
batch_size    = 32

class ReplayBuffer():
    def __init__(self):
        self.buffer = collections.deque(maxlen=buffer_limit)
    
    def put(self, transition):
        self.buffer.append(transition)
    
    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []
        
        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
               torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
               torch.tensor(done_mask_lst)
    
    def size(self):
        return len(self.buffer)

class Qnet(nn.Module):
    def __init__(self):
        super(Qnet, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    # 行动策略使用ε-greedy策略(off-policy的体现)
    def sample_action(self, obs, epsilon):
        #(batch_size, 2)
        out = self.forward(obs)
        # Q函数增加随机性的方式之一是ε-greedy策略，即在一定概率下随机选择动作，以探索更多可能的动作
        coin = random.random()
        if coin < epsilon:
            # 随机选择一个动作,在CartPole-v1中，动作有两个(0,1)
            return random.randint(0,1)
        else :
            # 选择具有最大 Q 值的动作
            return out.argmax().item()
            
def train(q, q_target, memory, optimizer):
    for i in range(10):
        s,a,r,s_prime,done_mask = memory.sample(batch_size)
        # 计算 Q(s,a)
        # (batch_size, 2)
        q_out = q(s)
        # (batch_size, 1) 获取第2维度的状态
        # (当前结果)
        q_a = q_out.gather(1,a)
        # 评价更新策略(off-policy的体现) 采用最大 Q 值
        max_q_prime = q_target(s_prime).max(1)[0].unsqueeze(1)
        # (参考结果)现状态做动作后拿到分数 = 当前奖励 + 下一步最好的预期分数(老师)
        target = r + gamma * max_q_prime * done_mask
        loss = F.smooth_l1_loss(q_a, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def main():
    env = gym.make('CartPole-v1', render_mode="human")
    # 这里生成两个网络，一个用于训练，一个用于预测
    q = Qnet()
    q_target = Qnet()
    q_target.load_state_dict(q.state_dict())
    memory = ReplayBuffer()

    print_interval = 20
    score = 0.0  
    optimizer = optim.Adam(q.parameters(), lr=learning_rate)

    for n_epi in range(10000):
        # 降低探索率阈值
        epsilon = max(0.01, 0.08 - 0.01*(n_epi/200)) #Linear annealing from 8% to 1%
        # (state_dim = 4)
        s, _ = env.reset()
        done = False
        env.render()
        while not done:
            # 根据探索率和ε-greedy策略选择动作
            a = q.sample_action(torch.from_numpy(s).float(), epsilon)
            s_prime, r, done, truncated, info = env.step(a)
            done_mask = 0.0 if done else 1.0
            # 保存动作
            memory.put((s,a,r/100.0,s_prime, done_mask))
            s = s_prime

            score += r
            if done:
                break
            
        if memory.size()>2000:
            train(q, q_target, memory, optimizer)

        if n_epi%print_interval==0 and n_epi!=0:
            # 每隔一段时间更新目标网络，这样可以使得q网络更加稳定
            q_target.load_state_dict(q.state_dict())
            print("n_episode :{}, score : {:.1f}, n_buffer : {}, eps : {:.1f}%".format(
                                                            n_epi, score/print_interval, memory.size(), epsilon*100))
            score = 0.0
    env.close()

if __name__ == '__main__':
    main()
