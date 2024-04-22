import numpy as np
import matplotlib.pyplot as plt
import math

np.random.seed(0)

def get_radar(pos_pred):
    vel_w = np.random.normal(0,5)   ## 시스템 오차 (속도)
    pos_w = np.random.normal(0,5)   ## 시스템 오차 (위치)
    vel   = 100 + vel_w             ## 고도 1km에서의 속도 
    alt   = 1000 + pos_w            ## 고도 (x_3)
    pos_pred = pos_pred + vel * dt  ## 위치 (x_1)
    v     = pos_pred * np.random.normal(0,0.05)  
    r     = math.sqrt(pos_pred**2 + alt**2) + v ## h(x)
    
    return r, pos_pred, vel, alt

def fx(x_esti):
    return A @ x_esti

def hx(x_pred):
    z_pred = np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    return np.array([z_pred])

def Hjacob_at(x_pred): ## H 행렬 연산 : 야코비안
    H[0][0] = x_pred[0] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    H[0][1] = 0
    H[0][2] = x_pred[2] / np.sqrt(x_pred[0]**2 + x_pred[2]**2)
    return H

dt       = 0.01
endTime  = 20
time     = np.arange(0, endTime, dt)
Nsamples = len(time)

position_save = np.zeros(Nsamples)
speed_save    = np.zeros(Nsamples)
altitude_save = np.zeros(Nsamples)
distance_save = np.zeros(Nsamples)
distance_esti = np.zeros(Nsamples)

A = np.eye(3) + np.array([[0, 1, 0],
                          [0, 0, 0],
                          [0, 0, 0]]) * dt
H = np.zeros((1,3))
Q = np.array([[0, 0, 0],
             [0, 0.001, 0],
             [0, 0, 0.001]])
R = np.array([[10]])

def extendedKalmanFilter(z_k, x_esti, P):
    x_pred = fx(x_esti)        ## 이전 추정값
    H      = Hjacob_at(x_pred) ## 이전 추정값을 통해 H 행렬 야코비안 연산
    
    ## (1) 추정값과 오차 공분산 예측
    P_pred = A @ P @ A.T + Q
    
    ## (2) 칼만 이득 계산
    K      = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    
    ## (3) 추정값 계산
    x_esti = x_pred + K @ (z_k - hx(x_pred))
    
    ## (4) 오차 공분산 계산
    P = P_pred - K @ H @ P_pred
    
    return x_esti, K, P

def distance_estimation(x_esti): ## 레이다로 측정한 실제 거리
    r_esti = math.sqrt(x_esti[0]**2 + x_esti[2]**2)
    return r_esti

pos_pred  = 0
r_esti    = None
x_0       = np.array([0, 90, 1100])
P_0       = 10 * np.eye(3)
x_esti, P = None, None

for i in range(Nsamples):
    z_k, pos_pred, vel, alt = get_radar(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, K, P = extendedKalmanFilter(z_k, x_esti, P)
        r_esti = distance_estimation(x_esti)
    
    position_save[i] = x_esti[0]
    speed_save[i]    = x_esti[1]
    altitude_save[i] = x_esti[2]
    distance_save[i] = z_k
    distance_esti[i] = r_esti

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 20), gridspec_kw={'hspace':1.0})

plt.subplot(2, 1, 1)
plt.plot(time, position_save, 'r', label='Estimation (EKF)', markersize=0.1)
plt.legend(loc='upper right')
plt.title('Position: Estimation: (EKF)')
plt.xlabel('time [s]'); plt.ylabel('position [m]')

plt.subplot(2, 1, 2)
plt.plot(time, altitude_save, 'r', label='Estimation (EKF)', markersize=0.1)
plt.legend(loc='upper right')
plt.title('Altitude: Estimation: (EKF)')
plt.xlabel('time [s]'); plt.ylabel('altitude [m]')

plt.show()

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 20), gridspec_kw={'hspace':1.0})

plt.subplot(2, 1, 1)
plt.plot(time, speed_save, 'r', label='Estimation (EKF)', markersize=0.1)
plt.legend(loc='upper right')
plt.title('Speed: Estimation: (EKF)')
plt.xlabel('time [s]'); plt.ylabel('position [m/s]')

plt.subplot(2, 1, 2)
plt.plot(time, distance_save, 'r:', label='Measurement', markersize=0.03, alpha=0.4)
plt.plot(time, distance_esti, 'r', label='Estimation (EKF)', markersize=0.1)
plt.legend(loc='upper right')
plt.title('Distance: Measurement vs. Estimation)')
plt.xlabel('time [s]'); plt.ylabel('distance [m]')

plt.show()
