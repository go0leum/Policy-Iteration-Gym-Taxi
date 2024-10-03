import gymnasium as gym
from policyiteration import PolicyIteration

# environment 생성
env = gym.make('Taxi-v3', render_mode="ansi")

# policy iteration 생성 및 설정
policyIteration = PolicyIteration(env=env, discount_factor=0.95)

# policy iteration 반복
N = 5
for n in range(N):

    print("{}-1. Policy Evaluation".format(n+1))
    policyIteration.policy_evaluation(iter_num=100)

    print("{}-2. Policy improvement".format(n+1))
    policyIteration.policy_improvement()

    print()
    policyIteration.print_value()
    print()

# policy iteration 결과 출력
print("3. print result")
policyIteration.print_result()