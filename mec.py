import math, random
import numpy as np
import mxnet as mx
from mxnet import nd, autograd
from mxnet.gluon import nn
from collections import deque
import math
import argparse
from mxnet import init, gluon


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ue', type=int, default=5)
    parser.add_argument('--F', type=int, default=5)
    args = parser.parse_args()
    return args

EPSILON = 0.9
def action_choise(q_table1, q_table2, tc, ac):
    if np.random.uniform()<EPSILON:
        ra = []
        rf = []
        for i in range(q_table1.shape[2]):
            ra.append(np.argmax(q_table1[tc][ac][i]))
        
        for i in range(q_table1.shape[2]):
            if ra[i] == 1.:
                if np.argmax(q_table2[tc][ac][i])!=0:
                    rf.append(np.argmax(q_table2[tc][ac][i]))
                else:
                    rf.append(1)
            else:
                rf.append(0)
        return ra, rf
    else:
        ra = np.random.randint(2, size=q_table1.shape[2])
        rf = np.zeros(q_table1.shape[2])
        for i in range(ra.size):
            if ra[i] == 1. :
                rf[i] = F / sum(ra)
        return list(ra), list(rf)
    
def sum_cost(ra, rf, dn, bn, dist):
    tc = 0
    for i in range(len(ra)):
        if ra[i]==0.:
            tc += it*dn[i] / (f*1000)
            tc += ie* dn[i]*1000*1000*pow(10,-27)*pow(f*1000*1000*1000,2)
        else:
            tmp_rn = 1000*1000* W / sum(ra) 
            mw = pow(10, -174 / 10) * 0.001
            rn = tmp_rn * np.log10(1+pn*0.001*pow(dist[i],-3) / (tmp_rn * mw))
            tc += it * bn[i] * 1024 / rn + ie * pn*0.001*bn[i]*1024 / rn
            tc += it*dn[i] / (rf[i]*1000) + ie * dn[i]*1000*1000*pi*0.001 / (rf[i]*1000*1000*1000)
    return tc


# def learn(state,action,reward,obervation):
#     q_table[state][action]+=ALPHA*(reward+GAMMA*max(q_table[obervation])-q_table[state,action])
    
    
def learn(q_table1, q_table2, tc, ac, ra, rf, reward, next_tc, next_ac):
    ALPHA , GAMMA = 0.01, 0.8
#     print(tc, ac, ra, rf, reward, next_tc, next_ac)
    for i in range(q_table1.shape[2]):
        q_table1[tc][ac][i][ra[i]] += ALPHA*(reward+GAMMA*np.max(q_table1[next_tc, next_ac])-q_table1[tc, ac, i, ra[i]])
        q_table2[tc][ac][i][rf[i]] += ALPHA*(reward+GAMMA*np.max(q_table2[next_tc, next_ac])-q_table2[tc, ac, i, rf[i]])



if __name__ == '__main__':
    
    args = parse_args()
    print(args)
    W = 10 # MHz 带宽
    F = args.F  # Ghz/sec MEC 计算能力
    f = 1  # Ghz/sec 本地 计算能力
    num_ue = args.num_ue # ue的个数
    
    dist = np.random.random(size=num_ue) * 200     # 每个ue的距离基站
    bn = np.random.uniform(300, 500, size=num_ue)  # 输入量 kbits
    dn = np.random.uniform(900,1100, size=num_ue)  # 需要周期量 兆周期数 1Mhz = 1000khz = 1000 * 1000hz
#     tao = np.random.
    it , ie = 0.5, 0.5 # 权重
    pn , pi = 500, 100 # 传输功率， 闲置功率 mW


    # Full Local
    # 延迟+能耗
    cost_full_local = sum( it*dn/(f*1000) + ie* dn*1000*1000*pow(10,-27)*pow(f*1000*1000*1000,2) )
    print('Full_local ', cost_full_local)
    
    # Full Offload
    # cost_full_Offload = bn / 
    tmp_rn = 1000*1000* W / num_ue 
    mw = pow(10, -174 / 10) * 0.001
    rn = tmp_rn * np.log10(1+pn*0.001*pow(dist,-3) / (tmp_rn * mw))
#     print(rn)
#     x dbm = y mW
#     x/10 = lg(y)
    # 第一步延迟和能量损失
    t1 = sum(it * bn * 1024 / rn + ie * pn*0.001*bn*1024 / rn)
    # 第二步延迟和能量损失
    t2 = sum(it*dn / (F*1000/num_ue) + ie * dn*1000*1000*pi*0.001 / (F*1000*1000*1000/num_ue))
    print('Full Offload ', t1+t2)


    # Q-learning

    SCORE=0
    max_tc = 100
    q_table1 = np.zeros((max_tc, F+1, num_ue, 2), dtype=np.float32)
    q_table2 = np.zeros((max_tc, F+1, num_ue, F+1), dtype=np.float32)
    
    avgs = []
    qlearning = cost_full_local
    for eps in range(30):
        # 初始化
        ra = np.random.randint(2, size=num_ue)

        rf = np.zeros(num_ue)
        for i in range(ra.size):
            if ra[i] == 1. :
                rf[i] = F / sum(ra)

        tc = 0
        for i in range(ra.size):
            if ra[i]==0.:
                tc += it*dn[i] / (f*1000)
                tc += ie* dn[i]*1000*1000*pow(10,-27)*pow(f*1000*1000*1000,2)
            else:
                tmp_rn = 1000*1000* W / sum(ra) 
                mw = pow(10, -174 / 10) * 0.001
                rn = tmp_rn * np.log10(1+pn*0.001*pow(dist[i],-3) / (tmp_rn * mw))
                tc += it * bn[i] * 1024 / rn + ie * pn*0.001*bn[i]*1024 / rn
                tc += it*dn[i] / (rf[i]*1000) + ie * dn[i]*1000*1000*pi*0.001 / (rf[i]*1000*1000*1000)
        ac = 0

        cnt = 1
        avg = cost_full_local

        for i in range(int(100000/0.8)):
            ra, rf = action_choise(q_table1, q_table2, int(tc), int(ac))
            if sum(rf) > F :
                break
            next_tc = sum_cost(ra, rf, dn, bn, dist)
            qlearning = min(qlearning, next_tc)
            
            avg += next_tc
            cnt += 1
            reward = (cost_full_local - next_tc) / cost_full_local
            
            learn(q_table1, q_table2, int(tc), int(ac), ra, [int(i) for i in rf], reward, int(next_tc), int(F-sum(rf)) )
            tc , ac = next_tc, int(F-sum(rf))
        avgs.append(avg / cnt)
        print('train epoch %2d'%eps, 'avgSumCost = ', avg / cnt)
        
    print('Q-learning ', qlearning, avgs)
