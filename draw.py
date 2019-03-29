import matplotlib.pyplot as plt
import numpy as np

file = open('log.txt')
full_local = []
full_offload = []
for line in file:
    line = line[:-1].split(' ')
    line = [ float(i) for i in line]
    full_local.append(line[0])
    full_offload.append(line[1])

num_ue = [3, 4, 5, 6,7]
print(full_local)
print(full_offload)

plt.plot(num_ue, full_local, label='Full Local')
plt.plot(num_ue, full_offload, label='Full Offload')
plt.grid(True)

plt.xlabel('the number of UE')
plt.ylabel('sum cost')
plt.legend(loc='upper left')
plt.show()