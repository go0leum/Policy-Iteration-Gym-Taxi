import gymnasium as gym
from policyiteration import PolicyIteration

# policy iteration 1번,policy improvement 1번을 10번 반복
print("======iter:1======")
N = 10
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

# policy iteration 결과 출력
print("3. print result")
policyIteration.print_result()

# policy iteration 10번, policy improvement 1번
print("======iter:10======")
N = 10
# environment 생성
env = gym.make('Taxi-v3', render_mode="ansi")
# policy iteration 생성 및 설정
policyIteration = PolicyIteration(env=env, discount_factor=0.95)
for n in range(N):

    print("{}-1. Policy Evaluation".format(n+1))
    policyIteration.policy_evaluation(iter_num=10)

    policyIteration.print_value()

print("{}-2. Policy improvement".format(n+1))
policyIteration.policy_improvement()
# policy iteration 결과 출력
print("3. print result")
policyIteration.print_result()