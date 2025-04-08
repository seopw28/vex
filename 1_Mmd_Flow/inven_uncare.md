# Mermaid 다이어그램 예제

## 데이터 분석 워크플로우

```mermaid
flowchart TD
    A[Start Data Analysis] --> B[Data Collection]
    B --> C[Data Cleaning]
    C --> D[Exploratory Data Analysis]
    D --> E{Data Quality Check}
    E -->|Issues Found| C
    E -->|Quality OK| F[Feature Engineering]
    F --> G[Model Selection]
    G --> H[Model Training]
    H --> I[Model Evaluation]
    I --> J{Performance Acceptable?}
    J -->|No| K[Hyperparameter Tuning]
    K --> H
    J -->|Yes| L[Model Deployment]
    L --> M[Monitoring & Maintenance]
    M --> N[End]
```
