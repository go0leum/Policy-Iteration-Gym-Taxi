# Policy Iteration: Gymnasium Taxi

[Gymnasium-Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/)


## Introduction

-  과제 내용

    1. Policy Iteration Class 구현
    2. 구현한 클래스를 이용한 policy 학습 및 결과 출력(q2.py)
    3. Policy Evaluation과 Policy Improvement 반복 횟수에 따른 결과 비교(q3.py)
    4. Delta와 Policy Stable 조건 추가 구현 (q4.py)

-  상세 요구 사항

    1. Policy Iteration Class 구현
    2. Policy 학습 및 결과 출력
    3. Policy Evaluation과 Improvement 반복 횟수 비교
    4. Delta와 Policy Stable 조건 추가


## Policy Iteration Description

- 첫번째 단계 policy evaluation과 두번째 단계policy improvement로 이루어짐

    1. policy evaluation: Iterative Policy Evaluation

        - 가치함수를 초기화 한 후, Bellman Expectation Equation을 반복적으로 사용하여 테이블에 적어 놓은 값을 조금씩 업데이트해 나가는 방법론

    2. policy improvement: Greddy Policy Improvement

        - greedy policy: 먼 미래를 고려하지 않고 눈 앞의 이익을 최대화하는 선택을 취하는 방식


## Code Description

- `class PolicyIteration`: 주어진 environment에 대해 policy Iteration을 행하기 위한 class

    - `env:` 환경 초기화
    - `policy`: 정책을 상태 별로 action(up, down, left, right, pickup, drop off) 동일한 확률로 초기화
    - `value`: 가치함수 상태별로 초기화
    - `discount_factor`: 감가율 초기화

- `def policy evaluation(iter_num)`: 모든 상태에 대해 Bellman Expectation Equation을 계산하여 가치함수 업데이트

    - 주어진 `iter_num`만큼 반복하여 가치함수를 업데이트 한다.

- `def policy_improvement()` : 계산된 가치함수를 바탕으로 탐욕 정책을 수행하여 정책을 업데이트 한다.
    - 모든 행동에 대해 [보상 + (감가율 * 다음 상태 가치함수)] 계산한다.
    - 값이 최대가 되는 행동들에 대해 동일한 확률을 적용하고 해당하지 않은 행동을 0.0으로 한다.

- `def print_value(s)` : 주어진 행동 s에 대한 가치 함수 값을 출력한다.
- `def get_policy(s)` : 주어진 행동 s에 대한 정책을 반환한다.
- `def get_value(s)` : 주어진 행동 s에 대한 가치 함수값을 반환한다.
- `def print_result()` : 현재 환경에서 정책에 따른 결과를 출력한다.
    - `state`: 현재 상태
    - `reward` : 행동에 따른 보상
    - `action` : 수행한 행동

## Result

- q2.py
    
    policy iteration을 `N`번 수행하고 결과를 출력한다.
    
    - `N=5`, policy evaluation의 `iter_num=100` 으로 설정한 결과, 총 13 step을 진행하는 결과를 도출했다.
    - `print_value` 결과 값이 크게 변하지 않는 걸 보아 값이 거의 수렴했다는 것을 알 수 있다.

- q3.py
    
    policy iteration에서 policy evaluation 1번, policy improvement 1번을 10번 반복하는 것(1)과 policy evaluation 10번, policy improvement 1번하는 것(2)의 결과를 비교한다.
    
    - 첫번째 경우 학습이 잘 진행되어 좋은 결과를 나타내었다. 반면에 두번째 경우는 10번 이상 반복해도 좋은 결과를 얻을 수 없었다.
    - 첫번째의 경우는 가치함수가 업데이트된 후, 정책이 업데이트되고 정책을 바탕으로 가치함수를 더 나은 방향으로 업데이트한다. 반면에 두번째의 경우 정책은 마지막에 한번만 업데이트되어 정책이 가치함수의 업데이트에 영향을 미치지 못해 학습이 제대로 진행되지 않는다.

- q4.py & policyiteration_q4.py
    
    Delta (theta=0.00001) 와 Policy Stable 조건을 추가한다.

    - `Delta` : 업데이트 이전 가치함수의 값과 이후의 가치함수의 차이를 계산하고 그 값이 지정한 theta값보다 작아지면 policy evaluation 과정을 종료한다.
        - 업데이트 이후 가치함수 값 간의 차이가 적을 수록 수렴되었다는 결론을 얻을 수 있다.`Delta` 를 지정함으로써 수렴되었다고 판단한 조건을 설정할 수 있다.
    - `policy_stable` : policy improvement을 진행하기 전과 이후의 정책을 비교하여 변화가 없다면 정책이 안정되었다고 판단하고 더이상 policy iteration과정을 진행하지 않는다.
        - policy가 더이상 변하지 않는다면 충분히 학습되었다는 것을 의미하므로 불필요한 policy iteration을 하는 것을 방지할 수있다.
    - policy evaluation과 policy improvement가 10번 반복되었다. 가치함수가 수렴하였고 정책이 적절히 학습되어 좋은 결과를 얻을 수 있었다.

- q5.py

    학습에 적절한 Delta 값에 대한 의문이 생겨 Delta 값의 변화에 따른 Iteration 반복횟수 및 evaluation 반복횟수 변화 그리고 학습 결과를 비교해보는 실험을 추가로 진행했다.

    - 결과적으로 theta가 작아질 수록 iteration 반복횟수도 줄어드는 것을 알 수 있었다.
    - 더불어 policy evaluation의 반복 횟수 k는 평균적으로 감소하는 것을 알 수 있었다.
    - theta를 실험한 범위 내에서 학습 결과는 모두 괜찮았다. 이는 policy stable을 적용한 영향이 큰 것 같다.

    | theta | Iteration | average K(policy evaluation 평균 반복횟수) |
    | --- | --- | --- |
    | 1 | 16 | 10.875 |
    | 0.1 | 14 | 25.571 |
    | 0.01 | 13 | 39.07 |
    | 0.001 | 12 | 68.0 |
    | 0.0001 | 13 | 83.385 |
    | 0.00001 | 10 | 132.8 |