#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import chi2_contingency
import seaborn as sns

#%%
# 샘플 데이터 생성: 성별과 선호 색상의 관계
np.random.seed(42)  # 재현성을 위한 시드 설정

# 관찰된 빈도 데이터 생성
observed = np.array([
    [30, 15, 25],  # 남성이 선호하는 색상 (빨강, 파랑, 녹색)
    [20, 25, 35]   # 여성이 선호하는 색상 (빨강, 파랑, 녹색)
])

# 데이터프레임 생성
df = pd.DataFrame(observed, 
                 index=['남성', '여성'],
                 columns=['빨강', '파랑', '녹색'])

print("관찰된 빈도 테이블:")
print(df)

#%%
# 카이제곱 검정 수행
chi2, p, dof, expected = chi2_contingency(observed)

print("\n카이제곱 검정 결과:")
print(f"카이제곱 통계량: {chi2:.4f}")
print(f"p-값: {p:.4f}")
print(f"자유도: {dof}")

# 기대 빈도 테이블
expected_df = pd.DataFrame(expected, 
                          index=['남성', '여성'],
                          columns=['빨강', '파랑', '녹색'])
print("\n기대 빈도 테이블:")
print(expected_df)

#%%
# 결과 해석
alpha = 0.05
if p < alpha:
    print(f"\n결과: p-값({p:.4f})이 유의수준({alpha})보다 작으므로, 귀무가설을 기각합니다.")
    print("즉, 성별과 선호 색상 간에 유의한 관계가 있습니다.")
else:
    print(f"\n결과: p-값({p:.4f})이 유의수준({alpha})보다 크므로, 귀무가설을 기각하지 못합니다.")
    print("즉, 성별과 선호 색상 간에 유의한 관계가 없다고 볼 수 있습니다.")

#%%
# 한글 폰트 설정
import matplotlib.font_manager as fm
# 맑은 고딕 또는 다른 한글 지원 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
# 또는 다음과 같이 특정 폰트 파일 지정 가능
# font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우 기준 맑은 고딕 경로
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()

# Visualization
plt.figure(figsize=(12, 8))

# 1. Bar chart of observed frequencies
plt.subplot(2, 2, 1)
df.plot(kind='bar', ax=plt.gca())
plt.title('관찰된 빈도')
plt.ylabel('빈도')
plt.xticks(rotation=0)

# 2. Bar chart of expected frequencies
plt.subplot(2, 2, 2)
expected_df.plot(kind='bar', ax=plt.gca())
plt.title('기대 빈도')
plt.ylabel('빈도')
plt.xticks(rotation=0)

# 3. Heatmap of observed frequencies
plt.subplot(2, 2, 3)
sns.heatmap(df, annot=True, fmt='d', cmap='YlGnBu')
plt.title('관찰된 빈도 히트맵')

# 4. Difference between observed and expected frequencies
diff_df = df - expected_df
plt.subplot(2, 2, 4)
sns.heatmap(diff_df, annot=True, fmt='.2f', cmap='coolwarm', center=0)
plt.title('관찰 빈도 - 기대 빈도')

plt.tight_layout()
plt.savefig('chi_square_test_visualization.png', dpi=300)
plt.show()
#%%
# 추가 예제: 더 큰 데이터셋으로 카이제곱 검정
print("\n\n추가 예제: 교육 수준과 직업 유형의 관계")

# 더 큰 관찰 데이터 생성
education_job = np.array([
    [50, 30, 20, 10],  # 고졸 (사무직, 기술직, 서비스직, 전문직)
    [40, 50, 30, 40],  # 대졸 (사무직, 기술직, 서비스직, 전문직)
    [20, 30, 10, 60]   # 대학원졸 (사무직, 기술직, 서비스직, 전문직)
])

edu_job_df = pd.DataFrame(education_job,
                         index=['고졸', '대졸', '대학원졸'],
                         columns=['사무직', '기술직', '서비스직', '전문직'])

print("관찰된 빈도 테이블:")
print(edu_job_df)

#%%
# 카이제곱 검정 수행
chi2_2, p_2, dof_2, expected_2 = chi2_contingency(education_job)

print("\n카이제곱 검정 결과:")
print(f"카이제곱 통계량: {chi2_2:.4f}")
print(f"p-값: {p_2:.4f}")
print(f"자유도: {dof_2}")

# 결과 해석
if p_2 < alpha:
    print(f"\n결과: p-값({p_2:.4f})이 유의수준({alpha})보다 작으므로, 귀무가설을 기각합니다.")
    print("즉, 교육 수준과 직업 유형 간에 유의한 관계가 있습니다.")
else:
    print(f"\n결과: p-값({p_2:.4f})이 유의수준({alpha})보다 크므로, 귀무가설을 기각하지 못합니다.")
    print("즉, 교육 수준과 직업 유형 간에 유의한 관계가 없다고 볼 수 있습니다.")

#%%
# 두 번째 예제 시각화
plt.figure(figsize=(10, 8))

# 한글 폰트 설정
import matplotlib.font_manager as fm
# 맑은 고딕 또는 다른 한글 지원 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
# 또는 다음과 같이 특정 폰트 파일 지정 가능
# font_path = 'C:/Windows/Fonts/malgun.ttf'  # 윈도우 기준 맑은 고딕 경로
# font_prop = fm.FontProperties(fname=font_path)
# plt.rcParams['font.family'] = font_prop.get_name()

# 그래프 그리기
sns.heatmap(edu_job_df, annot=True, fmt='d', cmap='viridis')
plt.title('교육 수준과 직업 유형의 관계')
plt.savefig('chi_square_education_job.png', dpi=300)
plt.show()

# %%
