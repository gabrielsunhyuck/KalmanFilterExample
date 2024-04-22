######### 관성 항법 센서 #########

import numpy as np 
import matplotlib.pyplot as plt
from scipy import io

input = io.loadmat('/home/sun/KalmanFilter/CH11.ARS/getfromGyro/ArsGyro.mat')

def get_gyro(i):
    p = input['wx'][i][0]
    q = input['wy'][i][0]
    r = input['wz'][i][0]
    
    return p, q, r

def euler_gyro(phi, the, psi, p, q, r, dt):
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    cos_the = np.cos(the)
    tan_the = np.tan(the)
    phi     = phi + dt * (p + q * sin_phi * tan_the + r * cos_phi * tan_the)  # phi = 이전 phi + phi_dot * dt
    the     = the + dt * (q * cos_phi - r * sin_phi)                      # the = 이전 the + phi_dot * dt
    psi     = psi + dt * (q * (sin_phi/cos_the) + r * (cos_phi/cos_the))  # psi = 이전 psi + phi_dot * dt
    
    return phi, the, psi

Nsamples = 41500
dt       = 0.01
time     = np.arange(Nsamples) * dt

phi_save = np.zeros(Nsamples)
the_save = np.zeros(Nsamples)
psi_save = np.zeros(Nsamples)

phi, the, psi = 0, 0, 0
for i in range(Nsamples):
    p, q, r       = get_gyro(i)
    phi, the, psi = euler_gyro(phi, the, psi, p, q, r, dt)
    phi_save[i]   = np.rad2deg(phi) # np.삼각함수는 라디안 값을 취급하므로 각도의 값으로 바꿔서 저장하도록 함
    the_save[i]   = np.rad2deg(the)
    psi_save[i]   = np.rad2deg(psi)
    
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

plt.subplot(3, 1, 1)
plt.plot(time, phi_save, 'r', label='Roll ($\\phi$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Roll ($\\phi$)')
plt.xlabel('time [s]')
plt.ylabel('Roll ($\phi$) angle [deg]')

plt.subplot(3, 1, 2)
plt.plot(time, the_save, 'b', label='Pitch ($\\theta$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Pitch ($\\theta$)')
plt.xlabel('Time [sec]')
plt.ylabel('Pitch ($\\theta$) angle [deg]')

plt.subplot(3, 1, 3)
plt.plot(time, psi_save, 'g', label='Yaw ($\\psi$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Yaw ($\\psi$)')
plt.xlabel('Time [sec]')
plt.ylabel('Yaw ($\\psi$) angle [deg]')

plt.show()
