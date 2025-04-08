# 인과추론 주요 용어 정리

1. 인과관계 (Causality)
- 원인과 결과 사이의 직접적인 관계
- 단순 상관관계와 달리 한 변수가 다른 변수에 직접적인 영향을 미치는 관계

2. 처치효과 (Treatment Effect)  
- 처치(중재, 개입)로 인한 결과의 변화량
- 처치를 받은 그룹과 받지 않은 그룹의 결과 차이

3. 반사실 (Counterfactual)
- 실제로 일어나지 않은 가상의 상황
- "처치를 받지 않았다면 어떠했을까?"에 대한 가정

4. 선택 편향 (Selection Bias)
- 데이터 수집 과정에서 특정 그룹이 과대/과소 표집되는 현상
- 무작위 할당이 이뤄지지 않아 발생하는 편향

5. 교란변수 (Confounding Variable)
- 원인과 결과 모두에 영향을 미치는 제3의 변수
- 인과관계 추정을 왜곡시킬 수 있는 요인

6. 도구변수 (Instrumental Variable)
- 원인변수와는 관련있지만 결과변수와는 직접적 관련이 없는 변수
- 인과효과 추정에 활용되는 방법론적 도구

7. 무작위화 (Randomization)
- 처치집단 배정을 무작위로 수행하는 것
- 관찰되지 않은 교란요인의 영향을 제거

8. 매개효과 (Mediation Effect)
- 원인이 매개변수를 통해 결과에 영향을 미치는 간접효과
- 인과경로를 설명하는 메커니즘

9. 조절효과 (Moderation Effect)
- 원인과 결과의 관계 강도를 변화시키는 제3의 변수 효과
- 상호작용 효과라고도 함

10. 평균처치효과 (ATE, Average Treatment Effect)
- 전체 모집단에서의 평균적인 처치효과
- 처치집단과 통제집단의 평균 차이

11. 처치집단 평균처치효과 (ATT, Average Treatment Effect on Treated)
- 실제 처치를 받은 집단에서의 평균 처치효과
- 선택적 처치 상황에서 중요

12. 인과다이어그램 (Causal Diagram)
- 변수들 간의 인과관계를 시각화한 그래프
- DAG(Directed Acyclic Graph) 형태로 표현

13. 시간적 선행성 (Temporal Precedence)
- 원인이 결과보다 시간적으로 선행해야 함
- 인과관계의 필요조건 중 하나

14. 외적타당도 (External Validity)
- 연구결과를 다른 상황에 일반화할 수 있는 정도
- 연구의 일반화 가능성

15. 내적타당도 (Internal Validity)
- 연구에서 도출된 인과관계의 신뢰성
- 다른 설명 가능성을 배제할 수 있는 정도

16. 동시발생 편향 (Simultaneity Bias)
- 원인과 결과가 서로 영향을 주고받는 상황
- 인과관계 방향 설정의 어려움

17. 회귀불연속설계 (RDD, Regression Discontinuity Design)
- 처치 할당의 임계점을 활용한 준실험 설계
- 임계점 주변에서의 인과효과 추정

18. 이중차분법 (DID, Difference in Differences)
- 처치 전후, 집단 간 차이를 비교하는 방법
- 시간에 따른 변화와 집단 간 차이 활용

19. 성향점수매칭 (PSM, Propensity Score Matching)
- 처치받을 확률이 비슷한 개체들을 매칭
- 선택편향 통제 방법

20. 합성대조군 (Synthetic Control)
- 여러 통제단위의 가중평균으로 비교군 구성
- 단일 사례 연구에서 활용

21. 생존편향 (Survival Bias)
- 생존한 대상만 관찰되어 발생하는 편향
- 성공사례만 보고 실패사례 간과

22. 평행추세가정 (Parallel Trends Assumption)
- 처치가 없었다면 두 집단이 비슷한 추세를 보였을 것이란 가정
- 이중차분법의 핵심 가정

23. 단절적 시계열분석 (ITS, Interrupted Time Series)
- 처치 전후의 시계열 패턴 변화 분석
- 시간 추세를 고려한 인과효과 추정

24. 역인과관계 (Reverse Causality)
- 결과가 원인에 영향을 미치는 현상
- 인과관계 방향의 오류

25. 생태학적 오류 (Ecological Fallacy)
- 집단 수준의 관계를 개인 수준으로 일반화하는 오류
- 집계 데이터 해석의 주의점

26. 처치효과 이질성 (Treatment Effect Heterogeneity)
- 처치효과가 개인/집단별로 다르게 나타나는 현상
- 하위집단 분석의 중요성

27. 조건부독립성 (Conditional Independence)
- 통제변수를 고려했을 때 처치할당이 결과와 독립적
- 인과추론의 핵심 가정

28. 균형성 검정 (Balance Test)
- 처치집단과 통제집단의 사전 특성 비교
- 무작위화 성공 여부 확인

29. 플라시보 검정 (Placebo Test)
- 가짜 처치를 통한 인과관계 검증
- 추정된 효과의 신뢰성 확인

30. 민감도 분석 (Sensitivity Analysis)
- 가정이나 모형 변경에 따른 결과 변화 검토
- 결과의 강건성 확인
