import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ue', type=int, default=5)
    parser.add_argument('--F', type=int, default=5)
    args = parser.parse_args()
    return args


import numpy as np


class Enviroment():
    def __init__(self, W, num_ue, F, bn, dn, dist, f, pn, pi, it=0.5, ie=0.5):
        self.W, self.num_ue, self.F = W, num_ue, F
        self.bn, self.dn, self.dist = bn, dn, dist
        self.f, self.it, self.ie = f, it, ie
        self.pn, self.pi = pn, pi

        # W = 10  # MHz 带宽
        # F = args.F  # Ghz/sec MEC 计算能力
        # f = 1  # Ghz/sec 本地 计算能力
        # num_ue = args.num_ue  # ue的个数
        #
        # dist = np.random.random(size=num_ue) * 200  # 每个ue的距离基站
        # bn = np.random.uniform(300, 500, size=num_ue)  # 输入量 kbits
        # dn = np.random.uniform(900, 1100, size=num_ue)  # 需要周期量 兆周期数 1Mhz = 1000khz = 1000 * 1000hz
        # #     tao = np.random.
        # it, ie = 0.5, 0.5  # 权重
        # pn, pi = 500, 100  # 传输功率， 闲置功率 mW

    def get_Init_state(self):  # 随机初始化, 返回 tc, ac, ra, rf 消耗，剩余F，此时的ra, rf
        ra = np.random.randint(2, size=num_ue)
        # rf = np.random.randint(F, size=num_ue)
        rf = np.zeros(ra.size)
        for i in range(ra.size):
            if ra[i] == 1.0:
                rf[i] = self.F / sum(ra)
        tc = 0
        for i in range(ra.size):
            if ra[i] == 0.:
                tc += self.it * self.dn[i] / (self.f * 1000)
                tc += self.ie * self.dn[i] * 1000 * 1000 * pow(10, -27) * pow(self.f * 1000 * 1000 * 1000, 2)
            else:
                tmp_rn = 1000 * 1000 * self.W / sum(ra)
                mw = pow(10, -174 / 10) * 0.001
                rn = tmp_rn * np.log10(1 + self.pn * 0.001 * pow(self.dist[i], -3) / (tmp_rn * mw))
                tc += self.it * self.bn[i] * 1024 / rn + self.ie * self.pn * 0.001 * self.bn[i] * 1024 / rn
                tc += self.it * self.dn[i] / (rf[i] * 1000) + self.ie * self.dn[i] * 1000 * 1000 * self.pi * 0.001 / (
                        rf[i] * 1000 * 1000 * 1000)
        ac = 0
        return np.array([tc, ac]), ra, rf

    def all_local(self):  # 全部在本地执行的花费
        cost_full_local = sum(
            self.it * self.dn / (self.f * 1000) + self.ie * self.dn * 1000 * 1000 * pow(10, -27) * pow(
                self.f * 1000 * 1000 * 1000, 2))
        return cost_full_local

    def sum_cost(self, ra, rf):  # 返回总消耗
        pass

    def step(self, ra, rf):  # 返回下一个状态，以及奖励 next_state, reward, done
        done = False
        if sum(rf) > F:
            done = True
            return None, None, done
        else:
            tc = 0
            for i in range(ra.size):
                if ra[i] == 0.:
                    tc += self.it * self.dn[i] / (self.f * 1000)
                    tc += self.ie * self.dn[i] * 1000 * 1000 * pow(10, -27) * pow(self.f * 1000 * 1000 * 1000, 2)
                else:
                    tmp_rn = 1000 * 1000 * self.W / sum(ra)
                    mw = pow(10, -174 / 10) * 0.001
                    rn = tmp_rn * np.log10(1 + self.pn * 0.001 * pow(self.dist[i], -3) / (tmp_rn * mw))
                    tc += self.it * self.bn[i] * 1024 / rn + self.ie * self.pn * 0.001 * self.bn[i] * 1024 / rn
                    tc += self.it * self.dn[i] / (rf[i] * 1000) + self.ie * self.dn[
                        i] * 1000 * 1000 * self.pi * 0.001 / (
                                  rf[i] * 1000 * 1000 * 1000)
            rewald = (self.all_local() - tc) / self.all_local()
            return np.array([tc, self.F - sum(ra)]), rewald, done


from mxnet import nd, autograd, gluon, init
from mxnet.gluon import nn, loss as gloss

from collections import deque
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


def compute_td_loss(batch_size, net, loss_fn, replay_buffer):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    reward = nd.array(reward).reshape((-1, 1))
    done = nd.array(done).reshape((-1, 1))
    # print(state, type(state), state.shape)
    # print(next_state, type(next_state), next_state.shape)
    # print(action, type(action), len(action))
    # for i in action:
    #     print(i)
    #
    # print("reward"*20)
    # print(reward, type(reward), len(reward))
    # for i in reward:
    #     print(i)
    #
    # print(done, type(done), len(done))
    gamma = 0.99

    q_value = net(nd.array(state))
    next_q_value = net(nd.array(state))
    expected_q_value = reward + gamma * next_q_value * (1 - done)
    loss = loss_fn(q_value, expected_q_value)
    return loss


def net_action(Y):
    ra, rf = [], []
    for i in range(0, num_ue * 2, 2):
        ra.append(np.argmax(Y[0, i:i + 2]))
    for i in range(len(ra)):
        if ra[i] == 1:
            rf.append(np.argmax(Y[0, num_ue * 2 + i * (F + 1):num_ue * 2 + (i + 1) * (F + 1)]))
        else:
            rf.append(0)
    return np.array(ra), np.array(rf)


def train(num_ue, F):
    replay_buffer = ReplayBuffer(capacity=200)  # 实例化经验池
    env = env = Enviroment(W=10, num_ue=num_ue, F=F, bn=np.random.uniform(300, 500, size=num_ue),
                           dn=np.random.uniform(900, 1100, size=num_ue),
                           dist=np.random.uniform(size=num_ue) * 200,
                           f=1, it=0.5, ie=0.5, pn=500, pi=100)  # 实例化环境

    net = nn.Sequential()  # 建立网络
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(num_ue * 2 + num_ue * (F + 1)))
    net.initialize(init.Normal(sigma=0.001))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})
    batch_size = 32
    loss_fn = gluon.loss.L2Loss()

    state, _, _, = env.get_Init_state()
    print(state)
    best_state = state[0]

    for idx in range(100000):
        action_ra, action_rf = net_action(net(nd.array(state.reshape((1, -1)))).asnumpy())

        next_state, reward, done = env.step(action_ra, action_rf)

        if done:
            # 重新初始化
            # 由于刚开始数据较少，这里制造了一些next_state
            next_state, ra, rf, = env.get_Init_state()
            _, reward, _ = env.step(ra, rf)
            best_state = state[0]
            replay_buffer.push(state, (ra, rf), reward, next_state, False)
            # push(self, state, action, reward, next_state, done):

            state, _, _, = env.get_Init_state()

        else:
            # print(state, end=' ')
            best_state = state[0]
            replay_buffer.push(state, (action_ra, action_rf), reward, next_state, done)
            state = next_state

        if len(replay_buffer) > 100:
            with autograd.record():
                loss = compute_td_loss(batch_size=batch_size, net=net, loss_fn=loss_fn, replay_buffer=replay_buffer)
                loss.backward()
            trainer.step(batch_size)
    print(best_state)


if __name__ == '__main__':
    args = parse_args()
    print(args)
    num_ue = args.num_ue
    F = args.F
    train(num_ue=num_ue, F=F)
