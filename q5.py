import gymnasium as gym
from policyiteration_q4 import PolicyIteration

for theta in [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]:
    print("++++++++++++theta={}++++++++++++".format(theta))
    # environment 생성
    env = gym.make('Taxi-v3', render_mode="ansi")
    # policy iteration 생성 및 설정
    policyIteration = PolicyIteration(env=env, discount_factor=0.95)

    n = 1
    policy_stable = False

    while not policy_stable:
        
        print("{}-1. Policy Evaluation".format(n))
        policyIteration.policy_evaluation(theta = theta)

        print("{}-2. Policy improvement".format(n))
        policy_stable = policyIteration.policy_improvement()
        
        n+=1
        print()

    # policy iteration 결과 출력
    print("3. print result")
    policyIteration.print_result()