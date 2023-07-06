import numpy as np

# 전환율과 샘플 수 설정하기
conversionRates = [0.15, 0.04, 0.13, 0.11, 0.05]
N = 2000
d = len(conversionRates)

# 데이터셋 생성하기
X = np.zeros((N, d))
for i in range(N):
    for j in range(d):
        if np.random.rand() < conversionRates[j]:
            X[i][j] = 1

nPosReward = np.zeros(d) # 이긴 횟수
nNegReward = np.zeros(d) # 진 횟수

# 베타 분포를 통해 최고의 슬롯머신을 선택하고 승패를 업데이트함
for i in range(N):
    selected = 0
    maxRandom = 0

    for j in range(d):
        randomBeta = np.random.beta(nPosReward[j]+1, nNegReward[j]+1)
        if randomBeta > maxRandom:
            maxRandom = randomBeta
            selected = j

    if X[i][selected] == 1:
        nPosReward[selected] += 1
    else:
        nNegReward[selected] += 1

# 최고라고 생각하는 슬롯머신 표시하기
nSelected = nPosReward + nNegReward
for i in range(d):
    print('Machine number ' + str(i+1) + 'was selected ' + str(nSelected[i]) + ' times')
print('Conclusion: Best machine is machine number ' + str(np.argmax(nSelected) + 1))