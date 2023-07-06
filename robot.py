# 물류를 위한 AI - 창고에서 일하는 롸봣

import numpy as np


# Q-러닝을 위한 매개변수 gamma와 alpha를 설정
gamma = 0.75
alpha = 0.9


# Part 1 - 환경 구성

# 상태 정의
location_to_state = {'A' : 0,
                     'B' : 1,
                     'C' : 2,
                     'D' : 3,
                     'E' : 4,
                     'F' : 5,
                     'G' : 6,
                     'H' : 7,
                     'I' : 8,
                     'J' : 9,
                     'K' : 10,
                     'L' : 11}

# 행동 정의
actions = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

# 보상 정의
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])


# Part 2 - Q-러닝으로 AI 솔루션 구성

# Q-값 초기화
Q = np.array(np.zeros([12, 12]))

# Q-러닝 프로세스 구현
for i in range(1000):
    current_state = np.random.randint(0, 12)
    playable_actions = []
    for j in range(12):
        if R[current_state, j] > 0:
            playable_actions.append(j)
    next_state = np.random.choice(playable_actions)
    TD = R[current_state, next_state] + gamma * Q[next_state, np.argmax(Q[next_state, ])] - Q[current_state, next_state]
    Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD


print("Q-values: ")
print(Q.astype(int))


# Part 3 - 운영 시작

# 상태에서 위치로 매핑 생성
state_to_location = {state: location for location, state in location_to_state.items()}

# 최적 경로를 반환하는 최종 함수 생성
def route(starting_location, ending_location):
    route = [starting_location]
    next_location = starting_location
    while(next_location != ending_location):
        starting_state = location_to_state[starting_location]
        next_state = np.argmax(Q[starting_state,])
        next_location = state_to_location[next_state]
        route.append(next_location)
        starting_location = next_location
    return route

# 최종 경로 출력
print('Route:')
print(route('E', 'G'))