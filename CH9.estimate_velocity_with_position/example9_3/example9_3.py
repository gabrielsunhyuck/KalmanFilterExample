######### 칼만 필터 예제 9-3 #########

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

np.random.seed(0)

get_distance = io.loadmat('/home/sun/KalmanFilter/CH9.estimate_velocity_with_position/example9_3/SonarAlt.mat')

print(get_distance)

def get_sonar(i):
    d = get_distance['sonarAlt'][0][i]  # input_mat['sonaralt']: (1, 1501)
    return d

def KalmanFilter(x_esti, z_k, P):
    # (1) 추정값, 오차 공분산 예측
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
    # (2) 칼만 이득 계산
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    # (3) 추정값 계산
    x_esti = x_pred + K @ (z_k - H @ x_pred)
    # (4) 오차 공분산 계산
    P = P_pred - K @ H @ P_pred
    
    return x_esti, P, K

Nsamples = 500
endTime  = 10
dt       = 0.02
time     = np.arange(0,endTime,dt)

distance_measured = np.zeros(Nsamples)
distance_estimate = np.zeros(Nsamples)
velocity_measured = np.zeros(Nsamples)
velocity_estimate = np.zeros(Nsamples)

A = np.array([[1, dt],
              [0, 1]])
H = np.array([[1, 0]]) ## 측정값은 [거리]정보일 뿐(위치 정보에 노이즈가 섞였다는 의미)이므로, 위치에 곱해지는 상수값은 0으로 처리되는 행렬
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[10]])

x_0 = np.array([0, 20])
P_0 = 5 * np.eye(2)

distance = 0
x_esti   = None
P        = None

for i in range(Nsamples):
    z_k = get_sonar(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = KalmanFilter(x_esti, z_k, P)
        
    distance_estimate[i] = x_esti[0]
    distance_measured[i] = z_k
    velocity_estimate[i] = x_esti[1]

plt.plot(time, distance_estimate, 'r-', label='Distance_estimated')
plt.plot(time, distance_measured, 'ko', markersize = 1, label='Distance_measured')
plt.xlabel('time [s]')
plt.ylabel('distance [m]')
plt.title('Distance : Measurement vs. Estimation')
plt.legend(loc='lower right')
plt.show()

plt.plot(time, velocity_estimate, 'b-', label='Velocity_estimated')
plt.xlabel('time [s]')
plt.ylabel('velocity [m/s]')
plt.title('Velocity : Measurement vs. Estimation')
plt.legend(loc='lower right')
plt.show()



