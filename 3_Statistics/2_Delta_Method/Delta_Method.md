## Delta Method (Delta Method)

델타 메소드는 함수의 근사 분산을 계산하는 데 사용되는 통계적 기법입니다. 

이는 주로 비선형 함수의 분산을 추정할 때 유용합니다. 
Delta Method는 테일러 급수를 사용하여 함수의 근사치를 구하고, 이를 통해 함수의 분산을 추정합니다.

### Delta Method의 기본 개념

델타 메소드는 다음과 같은 상황에서 사용됩니다:
- 우리가 관심 있는 양이 어떤 확률 변수의 함수로 표현될 때
- 그 함수의 분산을 추정하고자 할 때

예를 들어, 확률 변수 X의 함수 g(X)의 분산을 추정하고자 한다고 가정해봅시다. 
Delta Method는 g(X)를 X의 테일러 급수로 근사하여 분산을 계산합니다.

### Delta Method의 수학적 표현

델타 메소드는 다음과 같은 수학적 단계를 따릅니다:
1. 함수 g(X)를 X의 테일러 급수로 근사합니다.
2. 근사된 함수의 분산을 계산합니다.

구체적으로, 확률 변수 X의 함수 g(X)의 분산은 다음과 같이 근사될 수 있습니다:
\[ Var(g(X)) \approx \left( \frac{\partial g}{\partial X} \right)^2 Var(X) \]

여기서 \(\frac{\partial g}{\partial X}\)는 g(X)에 대한 X의 편미분입니다.

### Delta Method의 예시

델타 메소드를 사용하여 두 그룹의 전환율 비율의 분산을 추정하는 예시를 살펴보겠습니다. 
두 그룹 A와 B의 전환율을 각각 p_A와 p_B라고 할 때,
전환율 비율 (Effect Size)은 p_B / p_A로 계산됩니다.


이때, Delta Method를 사용하여 전환율 비율의 분산을 추정할 수 있습니다:

1. 전환율 비율의 함수는 f(p_A, p_B) = p_B / p_A입니다.

2. 이 함수의 편미분을 계산합니다:
   - \(\frac{\partial f}{\partial p_A} = -\frac{p_B}{p_A^2}\)
   - \(\frac{\partial f}{\partial p_B} = \frac{1}{p_A}\)

3. Delta Method를 사용하여 분산을 근사합니다:
\[ Var(f) \approx \left( \frac{\partial f}{\partial p_A} \right)^2 Var(p_A) + \left( \frac{\partial f}{\partial p_B} \right)^2 Var(p_B) \]

이를 통해 전환율 비율의 분산을 추정할 수 있습니다.

### 결론

델타 메소드는 비선형 함수의 분산을 추정하는 데 유용한 통계적 기법입니다. 
이는 테일러 급수를 사용하여 함수의 근사치를 구하고, 
이를 통해 함수의 분산을 계산합니다. 
Delta Method는 다양한 통계적 분석에서 중요한 역할을 합니다.
