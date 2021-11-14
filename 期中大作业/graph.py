import matplotlib.pyplot as plt
import numpy as np

rewards_0 = []
rewards_2 = []
x_0 = []
x_2 = []

wordlist = []
with open('rewards_0.txt', 'r') as f:
    for line in f:
        wordlist.append(line.split())

f.close()

reward_list_0 = []
for i in range(200):
    reward_list_0.append(float(wordlist[i][2]))
    x_0.append(int(wordlist[i][1]))
# print(x_0)
# print(reward_list_0)

wordlist = []
with open('rewards_2.txt', 'r') as f:
    for line in f:
        wordlist.append(line.split())

f.close()

reward_list_2 = []
for i in range(200):
    reward_list_2.append(float(wordlist[i][2]))
    x_2.append(int(wordlist[i][1]))
# print(x_2)
# print(reward_list_2)

x_0 = np.array(x_0)
x_2 = np.array(x_2)
reward_list_0 = np.array(reward_list_0)
reward_list_2 = np.array(reward_list_2)
w = 10
avg_x_0 = np.convolve(x_0, np.ones(w), 'valid')/w
avg_x_2 = np.convolve(x_2, np.ones(w), 'valid')/w
avg_r_list_0 = np.convolve(reward_list_0, np.ones(w), 'valid')/w
avg_r_list_2 = np.convolve(reward_list_2, np.ones(w), 'valid')/w


plt.title('Rewards')
plt.xlabel('Steps')
plt.ylabel('Rewards')
plt.plot(avg_x_0, avg_r_list_0)
plt.plot(avg_x_2, avg_r_list_2)

plt.legend(['DQN', 'Dueling-DQN'])
plt.show()
