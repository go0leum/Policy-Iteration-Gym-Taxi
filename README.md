# Policy Iteration: Gymnasium Taxi

[Gymnasium-Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/)


## Introduction

-  과제 내용

    1. Policy Iteration Class 구현
    2. 구현한 클래스를 이용한 policy 학습 및 결과 출력
    3. Policy Evaluation과 Policy Improvement 반복 횟수에 따른 결과 비교
    4. Delta와 Policy Stable 조건 추가 구현 (보너스 문제)

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