import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np


# 定义状态空间
class StateSpace:
    def __init__(self, data):
        self.data = data
        self.n_gates = 10  # 停机位数量
        self.n_flights = len(data)  # 航班数量
        self.state_dim = self.n_gates + self.n_flights * 3 + 2  # 状态维度


    def get_state(self, t):

        state = np.zeros(self.state_dim)
        state[0] = t  # 当前时刻
        for i in range(self.n_gates):
            state[i + 1] = self._get_gate_status(i, t)  # 停机位占用情况
        for i in range(self.n_flights):
            flight_info = self._get_flight_info(i)
            state[self.n_gates + 1 + i * 3:self.n_gates + 1 + (i + 1) * 3] = flight_info  # 航班属性
        state[-2] = 0  # 天气情况(简化为0)
        state[-1] = 0  # 航班流量(简化为0)
        return state

    def _get_gate_status(self, gate_id, t):
        for flight in self.data:
            if flight['停机位'] == gate_id and flight['到达时间'] <= t <= flight['出发时间']:
                return 1  # 占用
        return 0  # 空闲

    def _get_flight_info(self, flight_id):

        flight = self.data[flight_id]
        # 假设到达时间和出发时间已经是数值类型
        # 机型可以预先映射为数值（这里简化处理，直接使用优先级代替）
        # 注意：实际应用中需要更合理的处理机型映射
        return [flight['到达时间'], flight['出发时间'], flight['优先级']]

# 定义动作空间
class ActionSpace:
    def __init__(self, n_gates):
        self.n_gates = n_gates
        self.action_dim = n_gates + 2

    def sample(self):
        return np.random.randint(0, self.action_dim)

    def decode_action(self, action, flight_id):
        if action < self.n_gates:
            return f"将航班{flight_id}分配至停机位{action}"
        elif action == self.n_gates:
            return f"调整航班{flight_id}的停靠时间"
        else:
            return f"拒绝航班{flight_id}的停靠请求"


# 定义奖励函数
def calculate_reward(state, action, next_state):
    reward = 0
    # 停机位利用率
    reward += np.mean(state[1:11]) * 0.5  # 假设有10个停机位
    # 航班延误时间 - 这部分可能需要根据实际逻辑调整
    # 我们这里跳过，因为需要更多的上下文来正确实现它

    # 根据动作计算奖励
    if action < 10:  # 假设动作小于10时对应于停机位分配
        # 循环遍历每个航班的信息
        for i in range(len(data)):
            flight_info_index = 11 + i * 3  # 航班信息开始的索引
            # 确保我们不会越界
            if flight_info_index + 2 < len(state):
                flight_priority = state[flight_info_index + 2]  # 假设优先级是每个航班信息的第三个元素
                if flight_priority > 0:  # 如果有优先级，增加奖励
                    reward += 0.2
            else:
                break

    #     # 添加港湾约束
    # gate_conflict_penalty = 0
    # if action < 10:  # 假设动作编号小于10对应于分配到某个停机位
    #     assigned_gate = action
    #     for flight in data:
    #         if flight['停机位'] == assigned_gate:
    #             # 如果有飞机已经被分配到这个停机位，检查时间是否有冲突
    #             if not (flight['出发时间'] < state[0] or flight['到达时间'] > state[0]):
    #                 # 如果时间有冲突，应用惩罚
    #                 gate_conflict_penalty -= 100
    #
    # reward += gate_conflict_penalty
    return reward

    # 其他奖励逻辑...

    return reward


# Actor-Critic网络
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim),
            nn.Softmax(dim=-1)
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value


# PPO算法
class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_epsilon):
        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # 注意这里修改为添加一个批次维度
        action_probs, _ = self.actor_critic(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        return action.item()

    def update(self, states, actions, rewards, next_states, dones):
        # 对 states, actions, rewards, next_states, dones 进行 numpy 数组转换
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # 计算状态值和优势函数
        _, state_values = self.actor_critic(states)
        _, next_state_values = self.actor_critic(next_states)
        expected_state_action_values = rewards + self.gamma * next_state_values * (1 - dones)
        advantages = expected_state_action_values.squeeze() - state_values.squeeze()

        # 计算旧策略和新策略的概率比
        old_action_probs, _ = self.actor_critic(states)
        old_action_log_probs = torch.log(old_action_probs.gather(1, actions.unsqueeze(1)))

        # 更新策略
        action_probs, state_values = self.actor_critic(states)
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)))
        ratio = torch.exp(action_log_probs - old_action_log_probs.detach())
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (state_values.squeeze() - expected_state_action_values.detach()).pow(2).mean()

        self.optimizer.zero_grad()
        (policy_loss + value_loss).backward()
        self.optimizer.step()


# 训练
def train(data, n_episodes, max_steps):
    state_space = StateSpace(data)
    action_space = ActionSpace(state_space.n_gates)
    ppo = PPO(state_space.state_dim, action_space.action_dim, lr=0.001, gamma=0.99, clip_epsilon=0.2)

    for episode in range(n_episodes):
        state = state_space.get_state(0)
        rewards = []

        for step in range(max_steps):
            action = ppo.select_action(state)
            next_state = state_space.get_state(step + 1)
            reward = calculate_reward(state, action, next_state)
            done = (step == max_steps - 1)

            ppo.update([state], [action], [reward], [next_state], [done])

            state = next_state
            rewards.append(reward)

            if done:
                break

        print(f"Episode {episode}: Total Reward = {sum(rewards)}")


# 测试
def test(data, max_steps):
    state_space = StateSpace(data)
    action_space = ActionSpace(state_space.n_gates)
    ppo = PPO(state_space.state_dim, action_space.action_dim, lr=0.001, gamma=0.99, clip_epsilon=0.2)

    state = state_space.get_state(0)
    total_reward = 0

    for step in range(max_steps):
        action = ppo.select_action(state)
        next_state = state_space.get_state(step + 1)
        reward = calculate_reward(state, action, next_state)
        total_reward += reward

        print(f"Step {step}: {action_space.decode_action(action, step)}")

        state = next_state

        if step == max_steps - 1:
            break

    print(f"Total Reward = {total_reward}")


# 主程序
if __name__ == "__main__":
    data = [
    {'航班号': 'CA1234', '到达时间': 10, '出发时间': 9, '机型': 'B737', '优先级': 1, '停机位': -1},
    {'航班号': 'MU5678', '到达时间': 15, '出发时间': 10, '机型': 'A320', '优先级': 0, '停机位': -1},
    # ...
    ]

    train(data, n_episodes=100, max_steps=50)
    test(data, max_steps=50)

    # 最小鲁棒性



    # 港湾约束









