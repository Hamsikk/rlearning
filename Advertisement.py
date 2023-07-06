import numpy as np
import matplotlib.pyplot as plt
import random


N = 10000
d = 9


# 시뮬레이션 내 환경 생성
conversion_rates = [0.05, 0.13, 0.09, 0.16, 0.11, 0.04, 0.20, 0.08, 0.01] # 9가지 전략의 전환율
X = np.array(np.zeros([N, d]))
for i in range(N):
    for j in range(d):
        if np.random.rand() <= conversion_rates[j]:
            X[i, j] = 1


# 무작위 선택과 톰슨 샘플링 구현
strategies_selected_rs = []     # 라운드마다 무작위 선택 알고리즘에 의해 선택된 전략을 포함한 리스트
strategies_selected_ts = []     # 라운드마다 톰슨 샘플링 AI 모델에 의해 선택된 전략을 포함한 리스트
total_reward_rs = 0             # 무작위 선택 알고리즘에 의해 라운드가 반복될 떄마다 누적된 보상의 총합
total_reward_ts = 0             # 톰슨 샘플링 AI 모델에 의해 라운드가 반복될 떄마다 누적된 보상의 총합
numbers_of_rewards_1 = [0] * d  # 9개 요소로 이루어진 리스트로 각 전략이 보상으로 1을 받은 횟수를 포함
numbers_of_rewards_0 = [0] * d  # 9개 요소로 이루어진 리스트로 각 전략이 보상으로 0을 받은 횟수를 포함

for n in range(0, N):
    # 무작위 선택
    strategy_rs = random.randrange(d)
    strategies_selected_rs.append(strategy_rs)
    reward_rs = X[n, strategy_rs]
    total_reward_rs = total_reward_rs + reward_rs

    # 톰슨 샘플링
    strategy_ts = 0
    max_random = 0
    for i in range (0, d):
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            strategy_ts = i
    reward_ts = X[n, strategy_ts]
    if reward_ts == 1:
        numbers_of_rewards_1[strategy_ts] = numbers_of_rewards_1[strategy_ts] + 1
    else:
        numbers_of_rewards_0[strategy_ts] = numbers_of_rewards_0[strategy_ts] + 1
    strategies_selected_ts.append(strategy_ts)
    total_reward_ts = total_reward_ts + reward_ts


# 상대 수익률 계산
relative_return = (total_reward_ts - total_reward_rs) / total_reward_rs * 100
print("Relative Return : {:.0f} %".format (relative_return))

plt.hist(strategies_selected_ts)
plt.title('Histogram of Selections')
plt.xlabel('strategy')
plt.ylabel('Number of times the strategy was selected')
plt.show()