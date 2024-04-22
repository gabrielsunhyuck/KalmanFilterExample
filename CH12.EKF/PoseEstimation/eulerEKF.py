import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import io

np.random.seed(0)

accel_input = io.loadmat('/home/sun/KalmanFilter/CH12.EKF/PoseEstimation/Data/ArsAccel.mat')
gyro_input  = io.loadmat('/home/sun/KalmanFilter/CH12.EKF/PoseEstimation/Data/ArsGyro.mat')

def get_Accel(i):
    ax = accel_input['fx'][i][0]
    ay = accel_input['fy'][i][0]
    az = accel_input['fz'][i][0]
    
    return ax, ay, az

def get_gyro(i):
    p = gyro_input['wx'][i][0]
    q = gyro_input['wy'][i][0]
    r = gyro_input['wz'][i][0]
    
    return p, q, r

def accel_convert_euler(ax, ay, az, phi, the, psi):
    g      = 9.8
    cosThe = np.cos(the)
    phi    = np.arcsin(-ay / (g*cosThe))
    the    = np.arcsin(ax  / g)
    psi    = psi
    
    return phi, the, psi

def A_matrix(x_esti):
    phi, the, psi = x_esti

    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    tanThe = np.tan(the)
    secThe = 1. / np.cos(the)

    A = np.zeros((3, 3))

    A[0][0] = q*cosPhi*tanThe - r*sinPhi*tanThe
    A[0][1] = q*sinPhi*secThe**2 + r*cosPhi*secThe**2
    A[0][2] = 0

    A[1][0] = -q*sinPhi - r*cosPhi
    A[1][1] = 0
    A[1][2] = 0

    A[2][0] = q*cosPhi*secThe - r*sinPhi*secThe
    A[2][1] = q*sinPhi*secThe*tanThe + r*cosPhi*secThe*tanThe
    A[2][2] = 0

    A = np.eye(3) + A * dt
    return A

def fx(x_esti):
    phi, the, psi = x_esti
    
    sinPhi = np.sin(phi)
    cosPhi = np.cos(phi)
    tanThe = np.tan(the)
    secThe = 1. /np.cos(the)
    
    x_dot    = np.zeros(3)
    x_dot[0] = p + q*sinPhi*tanThe +r*cosPhi*tanThe
    x_dot[1] = q*cosPhi - r*sinPhi
    x_dot[2] = q*sinPhi*secThe + r*cosPhi*secThe
    
    x_pred   = x_esti + x_dot*dt ## f(x) 오일러 적분
    
    return x_pred

def hx(x_pred):
    return H @ x_pred

A = np.zeros((3,3))
H = np.eye(3)
Q = np.array([[0.0001, 0, 0],
              [0, 0.0001, 0],
              [0,    0, 0.1]])
R = 10 * np.eye(3)

Nsamples = 41500
dt       = 0.01
time     = np.arange(Nsamples) * dt

phi_save = np.zeros(Nsamples)
the_save = np.zeros(Nsamples)
psi_save = np.zeros(Nsamples)

def EKF(z_k, x_esti, P):
    A      = A_matrix(x_esti)
    ## (1) 추정값과 오차 공분산 예측
    x_pred = fx(x_esti)
    P_pred = A @ P @ A.T + Q
    ## (2) 칼만 이득 계산
    K      = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    ## (3) 추정값 계산
    x_esti = x_pred + K @ (z_k - hx(x_pred))
    ## (4)  오차 공분산 계산
    P      = P_pred - K @ H @ P_pred
    
    return x_esti, P, K
    
phi, the, psi = 0, 0, 0
x_esti, P     = None, None
x_0           = np.zeros(3)  
P_0           = 10 * np.eye(3)
for i in range(Nsamples):
    p, q, r       = get_gyro(i)
    ax, ay, az    = get_Accel(i)
    phi, the, psi = accel_convert_euler(ax, ay, az, phi, the, psi)
    z_k           = np.array([phi, the, psi])
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = EKF(z_k, x_esti, P)
        
    phi_save[i] = np.rad2deg(x_esti[0])
    the_save[i] = np.rad2deg(x_esti[1])
    psi_save[i] = np.rad2deg(x_esti[2])
    
fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(15, 12), gridspec_kw={'hspace':1.0})

plt.subplot(2,1,1)
plt.plot(time, phi_save, 'r', label='roll{$\phi$} : estimation (EKF)', markersize=0.3)
plt.legend(loc='lower right')
plt.title('Roll ($\phi$): Estimation (EKF)')
plt.xlabel('Time [s]'); plt.ylabel('Roll ($\phi$)')

plt.subplot(2,1,2)
plt.plot(time, the_save, 'r', label='pitch{$\the$} : estimation (EKF)', markersize=0.3)
plt.legend(loc='lower right')
plt.title('Pitch ($\the$): Estimation (EKF)')
plt.xlabel('Time [s]'); plt.ylabel('Pitch ($\the$)')

plt.show()