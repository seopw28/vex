
#%%
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

def perform_pca(X, n_components=None):
    """
    주성분 분석(PCA)을 수행하는 함수
    
    Parameters:
    -----------
    X : array-like, shape (n_samples, n_features)
        PCA를 적용할 입력 데이터
    n_components : int, optional (default=None)
        축소할 차원 수. None일 경우 원본 차원 유지
        
    Returns:
    --------
    X_pca : array-like
        차원 축소된 데이터
    pca : PCA object
        학습된 PCA 객체
    """
    
    # 데이터 스케일링
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # PCA 객체 생성 및 학습
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # 설명된 분산 비율 출력
    explained_variance_ratio = pca.explained_variance_ratio_
    print("설명된 분산 비율:", explained_variance_ratio)
    print("누적 설명된 분산 비율:", np.cumsum(explained_variance_ratio))
    
    # 시각화
    plt.figure(figsize=(10, 5))
    
    # 스크리 플롯
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, 'bo-')
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    
    # 누적 분산 비율
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
             np.cumsum(explained_variance_ratio), 'ro-')
    plt.title('Cumulative Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    
    plt.tight_layout()
    plt.show()
    
    return X_pca, pca

#%%
# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터 생성
    X = np.random.rand(100, 5)  # 100개 샘플, 5개 특성
    
    # PCA 수행 (3차원으로 축소)
    X_reduced, pca_model = perform_pca(X, n_components=3)
    print("\n변환된 데이터 형태:", X_reduced.shape)

# %%
