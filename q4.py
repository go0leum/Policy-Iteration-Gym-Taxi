import gymnasium as gym
from policyiteration_q4 import PolicyIteration

# environment 생성
env = gym.make('Taxi-v3', render_mode="ansi")

# policy iteration 생성 및 설정
policyIteration = PolicyIteration(env=env, discount_factor=0.95)

n = 1
policy_stable = False
# policy stable할 때까지 policy iteration 반복
while not policy_stable:

    print("{}-1. Policy Evaluation".format(n))
    policyIteration.policy_evaluation(theta = 0.00001)

    print("{}-2. Policy improvement".format(n))
    policy_stable = policyIteration.policy_improvement()
    
    n+=1
    print()

# policy iteration 결과 출력
print("3. print result")
policyIteration.print_result()