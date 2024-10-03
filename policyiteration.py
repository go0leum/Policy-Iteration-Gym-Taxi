import gymnasium as gym
import numpy as np
import time

class PolicyIteration:

    def __init__(self, env, discount_factor):
        # environment 초기화
        self.env = env
        # up,down,left,right,pickup,drop off를 동일한 확률로 초기화
        self.policy = np.ones((self.env.observation_space.n, self.env.action_space.n))/self.env.action_space.n
        # 가치함수를 array로 초기화
        self.value = np.zeros(self.env.observation_space.n) 
        # 감가율
        self.discount_factor = discount_factor

    def policy_evaluation(self, iter_num, print_value=False):
        # 새로운 가치함수 저장을 위해 초기화
        new_value = np.copy(self.value)

        for i in range(iter_num):
            # 모든 상태에 대해 Bellman expectatin equation 계산
            for s in range(self.env.observation_space.n):
                v = 0.0
                for a in range(self.env.action_space.n):
                    (probaility, s_next, reward, terminate) = self.env.unwrapped.P[s][a][0]
                    v += self.get_policy(s)[a]*(reward+self.discount_factor*self.get_value(s_next))
                
                new_value[s] = v
            
            # iter 마다 가치함수 업데이트
            self.value = new_value

            if print_value:
                self.print_value()
        
    def policy_improvement(self):
        # 새로운 정책 저장을 위해 초기화
        new_policy = np.copy(self.policy)

        # 모든 상태에 대해 greedy 정책 적용
        for s in range(self.env.observation_space.n):
            v_max = -1000000000

            policy_s = np.zeros(self.env.action_space.n)
            max_index = []
            
            # 모든 행동에 대해 [보상 + (감가율 * 다음 상태 가치함수)] 계산
            for a in range(self.env.action_space.n):
                (probaility, s_next, reward, terminate) = self.env.unwrapped.P[s][a][0]
                
                v = reward + self.discount_factor*self.get_value(s_next)
                
                # 받을 보상이 최대인 행동 모두 추출
                if v_max == v:
                    max_index.append(a)
                if v_max < v:
                    v_max = v
                    max_index.clear()
                    max_index.append(a)

            # 행동 확률 계산
            prob = 1/len(max_index)

            for index in max_index:
                policy_s[index] = prob

            new_policy[s] = policy_s
        
        # 정책 업데이트 
        self.policy = new_policy
    
    # 현재 가치함수 출력
    def print_value(self):
        print(self.value)
    
    # 특정 상태에 대한 정책 반환
    def get_policy(self, state):
        return self.policy[state]
    
    # 특정 상태에 대한 가치함수 반환
    def get_value(self, state):
        return self.value[state]
    
    # 정책에 대한 결과 출력
    def print_result(self):
        # 환경 초기화
        (obs, info) = self.env.reset()
        terminated = False
        truncated = False
        step = 1

        # 종료상태가 될 때까지 정책에 따라 행동
        while not (terminated or truncated):
            # 탐욕정책에 따라 action 선택
            action = np.argmax(self.policy[obs])
            (next_obs, reward, terminated, truncated, info) = self.env.step(action)

            # action에 따른 environment 상황 출력
            print("==========step:{}==========".format(step))
            print(self.env.render())
            print("state: {}, action: {}, reward: {}".format(next_obs, action, reward))
            
            obs = next_obs
            step += 1
            
            time.sleep(0.5)


