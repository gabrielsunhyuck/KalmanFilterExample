######### 칼만 필터 예제 9-2 #########

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv

np.random.seed(0)

dt      = 0.1
endTime = 4

t        = np.arange(0, endTime, dt)
Nsamples = len(t)

def get_Pose_Velocity(i_t):
    v_k = np.random.normal(0, np.sqrt(10))  

    velocity = 80                         
    position = velocity * (i_t * dt)  # 속도 적분을 통한 위치 추정
    z_k = velocity + v_k              # 속도 측정값 (노이즈 포함)       
    
    return z_k, position

def kalmanFilter(x_esti, z_k, P):
    # (1) 추정값, 오차 공분산 예측
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
    # (2) 칼만 이득
    K      = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
    # (3) 추정값 계산
    x_esti = x_pred + K @ (z_k - H @ x_pred)
    # (4) 오차 공분산 계산
    P      = P_pred - K @ H @ P_pred
    
    return x_esti, P, K

A = np.array([[1, dt],
              [0, 1]])
H = np.array([[0, 1]]) ## 측정값은 [속도]정보일 뿐(위치 정보에 노이즈가 섞였다는 의미)이므로, 위치에 곱해지는 상수값은 0으로 처리되는 행렬
Q = np.array([[1, 0],
              [0, 3]])
R = np.array([[10]])

x_0 = np.array([0, 20])
P_0 = 5 * np.eye(2)  ## 오차 공분산은 센서에서 계속 계측됨으로 최신화되는 값 계속 적용하는 것이 필요

position = 0
x_esti   = None
P        = None


x_measured  = np.zeros(Nsamples)
x_estimated = np.zeros(Nsamples)
v_measured  = np.zeros(Nsamples)
v_estimated = np.zeros(Nsamples)

for i in range(Nsamples):
    z_k, position = get_Pose_Velocity(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = kalmanFilter(x_esti, z_k, P)
        
    # x_measured[i]  = z_k
    x_estimated[i] = x_esti[0]
    v_measured[i]  = z_k
    v_estimated[i] = x_esti[1]
    
# plt.plot(t, x_measured, 'ro', label="Position_measured")
plt.plot(t, x_estimated, 'r:', label="Position_estimated")
plt.xlabel('time [s]')
plt.ylabel('position [m]')
plt.title('Position : Estimation')
plt.legend(loc='lower right')
plt.show()

plt.plot(t, v_measured, 'bo', label="Velocity_measured")
plt.plot(t, v_estimated, 'b:', label="Velocity_estimated")
plt.title('Velocity : Measurement vs. Estimation')
plt.xlabel('time [s]')
plt.ylabel('velocity [m/s]')
plt.legend(loc='lower right')
plt.show()
