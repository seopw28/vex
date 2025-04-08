#%%
import numpy as np
from scipy import stats

# A/B 그룹 전환 데이터
conversions_A = 120
visitors_A = 2400
conversions_B = 150
visitors_B = 2500

# 전환율 계산
p_A = conversions_A / visitors_A
p_B = conversions_B / visitors_B

# 전환율 비율 (Effect Size)
rate_ratio = p_B / p_A

# 델타 메소드를 이용한 분산 근사
var_A = p_A * (1 - p_A) / visitors_A
var_B = p_B * (1 - p_B) / visitors_B

# delta method: f(p_A, p_B) = p_B / p_A
# 근사된 분산: Var(f) ≈ (∂f/∂p_A)^2 * Var(p_A) + (∂f/∂p_B)^2 * Var(p_B)
# ∂f/∂p_A = -p_B / p_A^2
# ∂f/∂p_B = 1 / p_A

grad_pA = -p_B / (p_A ** 2)
grad_pB = 1 / p_A

var_ratio = (grad_pA ** 2) * var_A + (grad_pB ** 2) * var_B
se_ratio = np.sqrt(var_ratio)

#%%
# T-검정 기반 z-score (가우시안 근사)
z_score = (rate_ratio - 1) / se_ratio
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))

# 출력
print(f"전환율 A: {p_A:.4f}, 전환율 B: {p_B:.4f}")
print(f"전환율 비율 (B/A): {rate_ratio:.4f}")
print(f"표준 오차 (SE): {se_ratio:.4f}")
print(f"z-score: {z_score:.4f}")
print(f"p-value: {p_value:.4f}")

if p_value < 0.05:
    print("✅ 유의미한 차이가 있습니다.")
else:
    print("❌ 통계적으로 유의미하지 않습니다.")

# %%
