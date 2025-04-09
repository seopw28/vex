# dbt 자주 쓰는 코드 정리

## 1. 모델 생성 매크로
{% macro generate_staging_model(table_name) %}
select
    *
from {{ ref(table_name) }}
{% endmacro %}

## 2. 모델 정의
{{ config(
    materialized='table'
) }}

select
    *
from {{ ref('your_source_table') }}

## 3. 인크리멘탈 모델
{{ config(
    materialized='incremental',
    unique_key='id'
) }}

{% if is_incremental() %}

select
    *
from {{ ref('your_source_table') }}
where updated_at > (select max(updated_at) from {{ this }})

{% else %}

select
    *
from {{ ref('your_source_table') }}

{% endif %}

## 4. 테스트 정의
{% test not_null(model, column_name) %}
select
    *
from {{ model }}
where {{ column_name }} is null
{% endtest %}

## 5. Snapshot Definition
{% snapshot your_snapshot %}
{{ config(
    target_schema='snapshots',
    unique_key='id',
    strategy='timestamp',
    updated_at='updated_at'
) }}

select
    *
from {{ ref('your_source_table') }}
{% endsnapshot %}

## 6. 변수 사용
{{ var('your_variable', 'default_value') }}

## 7. 조건부 로직
{% if target.name == 'prod' %}
    -- Production specific logic
{% else %}
    -- Non-production logic
{% endif %}

## 8. 문서화
version: 2

models:
  - name: your_model
    description: "This is a description of your model"
    columns:
      - name: id
        description: "The primary key"
      - name: name
        description: "The name of the entity"

## 9. 소스 정의
version: 2

sources:
  - name: your_source
    tables:
      - name: your_table

## 10. 관계 정의
version: 2

models:
  - name: your_model
    description: "This is a description of your model"
    columns:
      - name: id
        description: "The primary key"
        tests:
          - unique
          - not_null
      - name: related_id
        description: "Foreign key to another model"
        tests:
          - relationships:
              to: ref('related_model')
              field: id
