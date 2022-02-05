
import numpy as np
import matplotlib.pyplot as plt

# State-Action R Table
R = np.array([
    # 在表格中每一个位置，向上、下、右、左走的话得到的回报
    # up, down, right, left
    # from bottom to top, left to right, specified S_0, S_1, ..., S48
    [[-1, np.nan, -100, np.nan], [-1, -1, -1, np.nan], [-1, -1, -1, np.nan], [np.nan, -1, -1, np.nan]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, np.nan, np.nan, np.nan], [-1, -100, -1, -1], [-1, -1, -1, -1], [np.nan, -1, -1, -1]],
    [[np.nan, 10, np.nan, np.nan], [-1, 10, np.nan, -1], [-1, -1, np.nan, -1], [np.nan, -1, np.nan, -1]]
])

# State-Action Q Table
Q = np.zeros((12, 4, 4))

def initMatrixQ():
    global Q
    Q = np.zeros((12, 4, 4))

#获取当前位置的有效行动
def getValidAction(position):
    action_list = []
    R_position = R[position]
    for index in range(len(R_position)):
        if not np.isnan(R_position[index]):
            action_list.append(index)
    return np.array(action_list)
#获取当前位置的下一步行动
#ϵ−greedy algorithm 每次选择执行的行为是估计值最大的行为，小概率的情况下，随机选择其他的行为。
#它以ϵ的概率从所有的action中随机抽取一个，以1−ϵ的概率抽取at=maxaQ(St,a;ω)。
def getNextActionWithEpsilonGreedy(position, epsilon, returnMax=False):
    valid_action = getValidAction(position)
    if len(valid_action) == 0:
        return -1
    random_float = np.random.rand()
    if random_float < epsilon and not returnMax:
        # randomly choose action from valid_action
        return valid_action[np.random.randint(0, len(valid_action))]
    else:
        # choose a* action form Q
        return valid_action[np.argmax(Q[position][valid_action])]

#根据当前位置和下一步行动获取下一个位置
def getNextPosition(curPosition, action):
    if action == 0:
        nextPosition = (curPosition[0], curPosition[1] + 1)
    elif action == 1:
        nextPosition = (curPosition[0], curPosition[1] - 1)
    elif action == 2:
        nextPosition = (curPosition[0] + 1, curPosition[1])
    elif action == 3:
        nextPosition = (curPosition[0] - 1, curPosition[1])
    if nextPosition[1] == 0 and nextPosition[0] <= 10 and nextPosition[0] >= 1:
        nextPosition = (0, 0)
    if curPosition == (11, 0):
        nextPosition = (11, 0)
    return nextPosition

def Sarsa(maxIteration=500, epsilon=0.1, alpha=0.5, gamma=0.9):
    initMatrixQ()
    total_reward_list = []
    for iterIndex in range(maxIteration):
        state = (0, 0)
        total_reward = 0.0
        while state != (11, 0):
            action = getNextActionWithEpsilonGreedy(state, epsilon)
            nextState = getNextPosition(state, action)
            nextAction = getNextActionWithEpsilonGreedy(nextState, epsilon)
            Q[state][action] += alpha * (R[state][action] + gamma * Q[nextState][nextAction] - Q[state][action])
            total_reward += R[state][action]
            state = nextState
        total_reward_list.append(total_reward)
    return total_reward_list

def QLearning(maxIteration=500, epsilon=0.1, alpha=0.5, gamma=0.9):
    initMatrixQ()
    total_reward_list = []
    for iterIndex in range(maxIteration):
        state = (0, 0)
        total_reward = 0.0
        while state != (11, 0):
            action = getNextActionWithEpsilonGreedy(state, epsilon)
            nextState = getNextPosition(state, action)
            nextAction = getNextActionWithEpsilonGreedy(nextState, epsilon, returnMax=True)
            Q[state][action] += alpha * (R[state][action] + gamma * Q[nextState][nextAction] - Q[state][action])
            total_reward += R[state][action]
            state = nextState
        total_reward_list.append(total_reward)
    return total_reward_list

if __name__ == '__main__':
    max_iteration = 50
    alpha = 0.5
    gamma = 1
    epsilon = 0.1

    sarsa_result = Sarsa(max_iteration, alpha=alpha, gamma=gamma, epsilon=epsilon)
    qlearning_result = QLearning(max_iteration, alpha=alpha, gamma=gamma, epsilon=epsilon)

    plt.title('Sarsa and QLearning Result')
    plt.plot(range(max_iteration), sarsa_result, color='red', label='Sarsa')
    plt.plot(range(max_iteration), qlearning_result, color='yellow', label='QLearning')
    plt.legend()
    plt.xlabel('episode')
    plt.ylabel('total reward per episode')
    plt.savefig('result.jpg')

    plt.show()

    print(sarsa_result)