import pandas as pd
import numpy as np
from scipy.optimize import minimize
import os

# CSV 파일 경로 설정
file_path = r'C:\Users\asus\OneDrive\Desktop\AI\data_stock.csv'

# CSV 파일에서 데이터 읽기
data = pd.read_csv(file_path)

# 시가총액 비율 설정
market_caps = np.array([0.557787, 0.103971, 0.0338242], dtype=float)

# 공분산 행렬 설정
cov_matrix = np.array([
    [0.001368733, 0.000757664, -0.000565785],
    [0.000757664, 0.000675273, -0.000505252],
    [-0.000565785, -0.000505252, 0.000783171]
], dtype=float)

####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################

# 위험회피계수와 스케일링 파라미터
risk_aversion = 5.0
tau = 0.05

# 투자자의 주관적 전망 설정
investor_views = np.array([
    0.0942,  # KBWB에 대한 투자자의 기대수익률 전망
    0.1514,  # XLI에 대한 투자자의 기대수익률 전망
    -0.05   # REK에 대한 투자자의 기대수익률 전망
], dtype=float)

# 전망에 대한 확신도 설정
view_confidences = np.array([
    0.516,  # KBWB 전망에 대한 확신도
    0.761,  # XLI 전망에 대한 확신도
    0.723   # REK 전망에 대한 확신도
], dtype=float)

####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################
####################이 값 조정하면 될 듯??##################

# 균형기대수익률 계산
def calculate_equilibrium_returns(market_caps, cov_matrix, risk_aversion):
    return risk_aversion * np.dot(cov_matrix, market_caps)

# 블랙-리터만 모델 계산
def black_litterman(market_caps, cov_matrix, risk_aversion, investor_views, view_confidences, tau):
    n = len(market_caps)
    pi = calculate_equilibrium_returns(market_caps, cov_matrix, risk_aversion)
    
    P = np.eye(n)
    omega = np.diag(view_confidences)
    
    A = np.dot(np.dot(tau * cov_matrix, P.T), np.linalg.inv(np.dot(np.dot(P, tau * cov_matrix), P.T) + omega))
    bl_return = pi + np.dot(A, investor_views - np.dot(P, pi))
    
    bl_cov = cov_matrix + np.dot(np.dot(tau * cov_matrix, P.T), np.linalg.inv(np.dot(np.dot(P, tau * cov_matrix), P.T) + omega))
    
    return bl_return, bl_cov

# 포트폴리오 최적화 함수
def portfolio_optimization(returns, cov_matrix, risk_aversion):
    n = len(returns)
    
    def objective(weights):
        portfolio_return = np.sum(returns * weights)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        return -(portfolio_return - 0.5 * risk_aversion * portfolio_volatility**2)
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    
    result = minimize(objective, n*[1./n], method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x

# 메인 실행
bl_return, bl_cov = black_litterman(market_caps, cov_matrix, risk_aversion, investor_views, view_confidences, tau)
optimal_weights = portfolio_optimization(bl_return, bl_cov, risk_aversion)

# 기존 코드는 그대로 두고 결과 출력 부분만 수정
print("\n블랙-리터만 기대수익률:")
for i, ret in enumerate(['KBWB', 'XLI', 'REK']):
    print(f"{ret}: {bl_return[i]:.6f}")

print("\n최적 포트폴리오 비중:")
for i, weight in enumerate(['KBWB', 'XLI', 'REK']):
    print(f"{weight}: {optimal_weights[i]:.4f}")

# 추가 분석 결과 출력
print("\n각 자산의 연평균 기대수익률:")
annual_returns = ((1 + bl_return) ** 52) - 1
for i, ret in enumerate(['KBWB', 'XLI', 'REK']):
    print(f"{ret}: {annual_returns[i]:.4%}")

# 펀드의 연평균 기대수익률 계산
fund_annual_return = np.sum(optimal_weights * annual_returns)
print(f"\n펀드의 연평균 기대수익률: {fund_annual_return:.4%}")

# 연환산 공분산 행렬
annual_cov_matrix = cov_matrix * 52

# 펀드의 연환산 표준편차와 분산
fund_annual_stddev = np.sqrt(np.dot(optimal_weights, np.dot(annual_cov_matrix, optimal_weights)))
fund_annual_variance = fund_annual_stddev ** 2

print(f"\n펀드의 연환산 표준편차: {fund_annual_stddev:.4%}")
print(f"펀드의 연환산 분산: {fund_annual_variance:.6f}")
