import gymnasium as gym
from policyiteration import PolicyIteration

# policy iteration 1번 반복
print("======iter:1======")
N = 100
# environment 생성
env = gym.make('Taxi-v3', render_mode="ansi")
# policy iteration 생성 및 설정
policyIteration = PolicyIteration(env=env, discount_factor=0.95)
for n in range(N):

    print("{}-1. Policy Evaluation".format(n+1))
    policyIteration.policy_evaluation(iter_num=1)

    print("{}-2. Policy improvement".format(n+1))
    policyIteration.policy_improvement()

    policyIteration.print_value()

# policy iteration 10번 반복
print("======iter:10======")
N = 100
# environment 생성
env = gym.make('Taxi-v3', render_mode="ansi")
# policy iteration 생성 및 설정
policyIteration = PolicyIteration(env=env, discount_factor=0.95)
for n in range(N):

    print("{}-1. Policy Evaluation".format(n+1))
    policyIteration.policy_evaluation(iter_num=10)

    print("{}-2. Policy improvement".format(n+1))
    policyIteration.policy_improvement()

    policyIteration.print_value()