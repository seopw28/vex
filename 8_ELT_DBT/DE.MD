# 데이터 엔지니어가 알아야 할 기본 개념 30가지

## 데이터 아키텍처 및 설계
1. **데이터 웨어하우스(Data Warehouse)**: 기업의 모든 데이터를 저장하고 분석하기 위한 중앙 저장소
2. **데이터 레이크(Data Lake)**: 구조화/비구조화된 모든 형태의 데이터를 원시 형태로 저장하는 저장소
3. **데이터 마트(Data Mart)**: 특정 부서나 기능을 위한 데이터 웨어하우스의 하위 집합
4. **OLTP vs OLAP**: 트랜잭션 처리 시스템과 분석 처리 시스템의 차이점
5. **스키마 설계**: Star, Snowflake, Galaxy 등 데이터 모델링 방법론

## 데이터 통합 및 ETL/ELT
6. **ETL(Extract, Transform, Load)**: 데이터 추출, 변환, 적재 프로세스
7. **ELT(Extract, Load, Transform)**: 데이터 추출, 적재 후 변환하는 현대적 접근법
8. **CDC(Change Data Capture)**: 소스 데이터의 변경사항을 식별하고 추적하는 기술
9. **데이터 파이프라인**: 데이터 소스에서 목적지까지의 자동화된 데이터 흐름
10. **배치 처리 vs 스트림 처리**: 데이터 처리의 두 가지 주요 패러다임

## 데이터베이스 및 저장 기술
11. **관계형 데이터베이스(RDBMS)**: SQL 기반 데이터베이스 (MySQL, PostgreSQL 등)
12. **NoSQL 데이터베이스**: 비관계형 데이터베이스 (MongoDB, Cassandra, Redis 등)
13. **분산 파일 시스템**: HDFS, S3와 같은 대용량 데이터 저장 시스템
14. **데이터 파티셔닝**: 대용량 데이터를 관리하기 위한 분할 전략
15. **인덱싱**: 데이터 검색 성능을 향상시키는 기술

## 빅데이터 기술
16. **하둡 에코시스템**: HDFS, MapReduce, YARN 등 빅데이터 처리 프레임워크
17. **스파크(Spark)**: 인메모리 기반 분산 데이터 처리 엔진
18. **카프카(Kafka)**: 실시간 데이터 스트리밍 플랫폼
19. **데이터 직렬화**: Avro, Parquet, ORC 등 효율적인 데이터 저장 형식
20. **MPP(Massive Parallel Processing)**: Redshift, Snowflake 등의 병렬 처리 데이터베이스

## 데이터 품질 및 거버넌스
21. **데이터 품질 관리**: 정확성, 완전성, 일관성, 적시성 등 데이터 품질 측정 및 관리
22. **메타데이터 관리**: 데이터에 대한 데이터를 체계적으로 관리하는 방법
23. **데이터 리니지(Lineage)**: 데이터의 출처와 변환 과정을 추적하는 기술
24. **마스터 데이터 관리(MDM)**: 핵심 비즈니스 데이터의 통합 관리
25. **데이터 카탈로그**: 조직 내 모든 데이터 자산을 문서화하고 검색 가능하게 하는 도구

## 인프라 및 운영
26. **클라우드 데이터 서비스**: AWS, GCP, Azure 등의 클라우드 기반 데이터 서비스
27. **컨테이너화 및 오케스트레이션**: Docker, Kubernetes를 활용한 데이터 애플리케이션 배포
28. **IaC(Infrastructure as Code)**: Terraform, CloudFormation 등을 통한 인프라 자동화
29. **CI/CD 파이프라인**: 데이터 파이프라인의 지속적 통합 및 배포
30. **데이터 보안 및 규정 준수**: 암호화, 접근 제어, GDPR, CCPA 등 데이터 보호 규정
