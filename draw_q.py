import matplotlib.pyplot as plt
import numpy as np

file = open('log_q.txt')
full_local, full_offload, qlearn = [], [], []

for line in file:
	if 'Full_local' in line:
		full_local.append(float(line[12:]))
	if 'Full Offload' in line:
		full_offload.append(float(line[14:]))
	if 'Q-learning' in line:
		line = line.split(' ')
		qlearn.append(float(line[2]))
# print(full_local)
# print(full_offload)
# print(qlearn)

num_ue = [3, 4, 5, 6, 7]
plt.plot(num_ue, full_local, '^-', label='Full Local')
plt.plot(num_ue, full_offload, 'v-', label='Full Offload')
plt.plot(num_ue, qlearn, 'o-', label='Q-learning')
plt.grid(True)

plt.xlabel('The number of UE')
plt.ylabel('Sum Cost')
plt.legend(loc='upper left')
plt.show()
