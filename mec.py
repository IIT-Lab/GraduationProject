import numpy as np
import mxnet as mx
from mxnet import nd, autograd
import math
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-ue', type=int, default=5)
    parser.add_argument('--F', type=int, default=5)
    args = parser.parse_args()
    return args

# data

if __name__ == '__main__':
    
    args = parse_args()
    
    W = 10 # MHz 带宽
    F = args.F  # Ghz/sec MEC 计算能力
    f = 1  # Ghz/sec 本地 计算能力
    num_ue = args.num_ue # ue的个数
    
    dist = np.random.random(size=num_ue) * 200     # 每个ue的距离基站
    bn = np.random.uniform(300, 500, size=num_ue)  # 输入量 kbits
    dn = np.random.uniform(900,1100, size=num_ue)  # 需要周期量 兆周期数
    it , ie = 0.5, 0.5 # 权重
    pn , pi = 500, 100 # 传输功率， 闲置功率 mW


    # Full Local
    # 延迟+能耗
    cost_full_local = sum( it*dn/(f*1024) + ie* dn*pow(10,-27)*pow(f*1024*1024*1024*1024,2) )
    print(cost_full_local)
    # Full Offload
    # cost_full_Offload = bn / 
    # rn = W / num_ue * math.log10(1+)