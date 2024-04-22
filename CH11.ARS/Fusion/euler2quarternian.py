#### euler2quarterian >> sensor fusion ####

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy import io

input_gyro_mat = io.loadmat('/home/sun/KalmanFilter/CH11.ARS/Fusion/ArsGyro.mat')
input_accel_mat = io.loadmat('/home/sun/KalmanFilter/CH11.ARS/Fusion/ArsAccel.mat')

def get_gyro(i):
    p = input_gyro_mat['wx'][i][0] 
    q = input_gyro_mat['wy'][i][0] 
    r = input_gyro_mat['wz'][i][0]     
    return p, q, r

def get_accel(i):
    ax = input_accel_mat['fx'][i][0] 
    ay = input_accel_mat['fy'][i][0] 
    az = input_accel_mat['fz'][i][0] 
    return ax, ay, az

def systemModel(p, q, r, dt):
    A = np.eye(4) + 0.5 * dt * np.array([[0, -p, -q, -r],
                                         [p,  0,  r, -q],
                                         [q, -r,  0,  p],
                                         [r,  q, -p,  0]])
    return A

def euler_accel(ax, ay, az, phi, the, psi):
    g       = 9.8 # [m/s^2]
    cos_the = np.cos(the)
    phi     = np.arcsin(-ay / (g * cos_the))
    the     = np.arcsin(ax / g)
    psi     = psi

    return phi, the, psi

def euler2Quarternion(phi, the, psi): ### 오일러 >> 쿼터니언 변환 함수 참고
    sin_phi = np.sin(phi/2); sin_the = np.sin(the/2); sin_psi = np.sin(psi/2)
    cos_phi = np.cos(phi/2); cos_the = np.cos(the/2); cos_psi = np.cos(psi/2)
    
    q       = np.array([cos_phi * cos_the * cos_psi + sin_phi * sin_the * sin_psi,
                        sin_phi * cos_the * cos_psi - cos_phi * sin_the * sin_psi,
                        cos_phi * sin_the * cos_psi + sin_phi * cos_the * sin_psi,
                        cos_phi * cos_the * sin_psi - sin_phi * sin_the * cos_psi])
    return q

def quaraternion2Euler(q):  ### 쿼터니언 >> 오일러 변환 함수 참고
    # q = q.ravel()
    phi_esti = np.arctan2(2 * (q[2]*q[3] + q[0]*q[1]), 1 - 2 * (q[1]**2 + q[2]**2))
    the_esti = -np.arcsin(2 * (q[1]*q[3] - q[0]*q[2]))
    psi_esti = np.arctan2(2 * (q[1]*q[2] + q[0]*q[3]), 1 - 2 * (q[2]**2 + q[3]**2))
    return phi_esti, the_esti, psi_esti

def KalmanFilter(z_k, x_esti, P):
    # (1) 추정값과 오차 공분산 예측
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
    # (2) 칼만 이득 계산
    K = P_pred @ H.T @ inv(H @ P_pred @ H.T + R)
    # (3) 추정값 계산
    x_esti = x_pred + K @ (z_k - H @ x_pred)
    # (4) 오차 공분산 계산
    P = P_pred - K @ H @ P_pred
    
    return x_esti, P

Nsamples = 41500
dt       = 0.01

H        = np.eye(4)
Q        = 0.0001 * np.eye(4)
R        = 10 * np.eye(4)

x_0      = np.array([1, 0, 0, 0])
P_0      = np.eye(4)

time          = np.arange(Nsamples) * dt
phi_estimated = np.zeros(Nsamples)
the_estimated = np.zeros(Nsamples)
psi_estimated = np.zeros(Nsamples)

phi, the, psi = 0, 0, 0
x_esti, A, P  = None, None, None
for i in range(Nsamples):
    p, q, r       = get_gyro(i)
    A             = systemModel(p, q, r, 0.01)
    ax, ay, az    = get_accel(i)
    phi, the, psi = euler_accel(ax, ay, az, phi, the, psi)
    z_k           = euler2Quarternion(phi, the, psi)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P = KalmanFilter(z_k, x_esti, P)
        
    phi_esti, the_esti, psi_esti = quaraternion2Euler(x_esti)
    
    phi_estimated[i] = np.rad2deg(phi_esti)
    the_estimated[i] = np.rad2deg(the_esti)
    psi_estimated[i] = np.rad2deg(psi_esti)
    
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20), gridspec_kw={'hspace': 1.0})

plt.subplot(3, 1, 1)
plt.plot(time, phi_estimated, 'r', label='Roll ($\phi$): Estimation (KF)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Roll ($\phi$): Estimation (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Roll ($\phi$) angle [deg]')

plt.subplot(3, 1, 2)
plt.plot(time, the_estimated, 'b', label='Pitch ($\\theta$): Estimation (KF)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Pitch ($\\theta$): Estimation (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch ($\\theta$) angle [deg]')

plt.subplot(3, 1, 3)
plt.plot(time, psi_estimated, 'g', label='Yaw ($\psi$): Estimation (KF)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Yaw ($\psi$): Estimation (KF)')
plt.xlabel('Time [sec]')
plt.ylabel('Yaw ($\psi$) angle [deg]')

plt.show()