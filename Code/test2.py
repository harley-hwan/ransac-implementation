import numpy as np
import pandas as pd

# CSV 파일 읽기
df = pd.read_csv('data_4.csv')

# 첫 번째 열을 numpy 배열로 변환
data = df.iloc[:, 0].to_numpy()

n = len(data)

# 칼만 필터 초기화
Q = 1e-5  # 프로세스 잡음 공분산
R = 0.1**2  # 측정 잡음 공분산
xhat = np.zeros(n)  # 추정치를 0으로 초기화
P = np.zeros(n)  # 오차 공분산을 0으로 초기화
xhatminus = np.zeros(n)  # 추정치의 사전 업데이트를 0으로 초기화
Pminus = np.zeros(n)  # 오차 공분산의 사전 업데이트를 0으로 초기화
K = np.zeros(n)  # 칼만 이득을 0으로 초기화

xhat[0] = 0.0
P[0] = 1.0

for k in range(1, n):
    # 시간 업데이트(예측)
    xhatminus[k] = xhat[k-1]
    Pminus[k] = P[k-1] + Q

    # 측정 업데이트(수정)
    K[k] = Pminus[k] / (Pminus[k] + R)
    xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
    P[k] = (1 - K[k]) * Pminus[k]

print("Filtered data:\n", xhat)

# numpy 배열을 pandas DataFrame으로 변환
df_output = pd.DataFrame(xhat, columns=['Filtered_Data'])

# DataFrame을 csv 파일로 저장
df_output.to_csv('output_4.csv', index=False)


