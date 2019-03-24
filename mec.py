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
    dn = np.random.uniform(900,1100, size=num_ue)  # 需要周期量 兆周期数 1Mhz = 1000khz = 1000 * 1000hz
    it , ie = 0.5, 0.5 # 权重
    pn , pi = 500, 100 # 传输功率， 闲置功率 mW


    # Full Local
    # 延迟+能耗
    cost_full_local = sum( it*dn/(f*1000) + ie* dn*1000*1000*pow(10,-27)*pow(f*1000*1000*1000,2) )
    print(cost_full_local)
    
    # Full Offload
    # cost_full_Offload = bn / 
    tmp_rn = 1000*1000* W / num_ue 
    mw = pow(10, -174 / 10) * 0.001
    rn = tmp_rn * np.log10(1+pn*0.001*pow(dn,-3) / (tmp_rn * mw))
#     print(rn)
#     x dbm = y mW
#     x/10 = lg(y)
    # 第一步延迟和能量损失
    t1 = sum(it * bn * 1024 / rn + ie * pn*0.001*bn*1024 / rn)
    # 第二步延迟和能量损失
    t2 = sum(it*dn / (F*1000/num_ue) + ie * dn*1000*1000*pi*0.001 / (F*1000*1000*1000/num_ue))
    print(t1+t2)
    
    
    