####### 이동평균 필터 ########

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input_sonar = io.loadmat('/home/sun/KalmanFilter/source/2.MovAvgFilter/SonarAlt.mat')
## input_sonar은 딕셔너리 형태로 받아온다. > sonarAlt 키에 해당하는 값을 가지고 올 필요 있음
## io.loadmat('~매트랩 파일 경로/매트랩 파일') : 매트랩 파일의 데이터를 파이썬으로 끌고 옴

def sonar(i):
    sonar_distance = input_sonar['sonarAlt'][0][i] 
    # 'sonarAlt':sonar 거리 데이터(value)를 담고 있는 key
    # [0] : 매트랩 데이터의 행 번호(파이썬은 0부터 시작, 매트랩은 1부터 시작)
    # [1] : 매트랩 데이터의 열 번호(파이썬은 0부터 시작, 매트랩은 1부터 시작)
    return sonar_distance

n         = 10   # [이동평균 데이터 개수]
Nsamples  = 500  # 
end_Time  = 10   # [s]
dt        = 0.02 # [s]
time      = np.arange(0, end_Time, dt)
x_measure = np.zeros(Nsamples)
x_average = np.zeros(Nsamples)

for i in range(Nsamples):
    x = sonar(i)     # 계측된 데이터
    x_measure[i] = x # 계측된 데이터 numpy 행렬 저장 > plot을 위해
    if i == 0:       # 초기값 설정
        x_avg = x
        x_n = np.full(n, x)
    else:
        x_n[:-1] = x_n[1:]   # for구문 내에서 이동평균 필터 배열 한 칸씩 다음으로 이동
                             # x_n[:-1] : 마지막 데이터를 제거한 후의 리스트 배열
                             # x_n[1:]  : 첫번째 데이터를 제거한 후의 리스트 배열
        x_n[-1] = x          # for구문 내에서 새로운 측정값을 배열의 마지막에 저장
        x_avg = np.mean(x_n)
        
        x_average[i] = x_avg

plt.figure(figsize=(10,6))
plt.scatter(time, x_measure, label='measured distance', s=20, c='k', marker='o', alpha=0.5)
plt.plot(time, x_average, label='filtered distance', c='r')

plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('MEASURED DISTANCE vs. MOVE AVERAGE FILTERED DISTANCE')
plt.legend()
plt.grid(True)
plt.show()
