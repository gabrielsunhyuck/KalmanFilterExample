######### 가속도계 센서 #########

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

input = io.loadmat('/home/sun/KalmanFilter/CH11.ARS/getfromARS/ArsAccel.mat')

def get_Accel(i):
    a_x = input['fx'][i][0]
    a_y = input['fy'][i][0]
    a_z = input['fz'][i][0]

    return a_x, a_y, a_z

def convert_Accel2Euler(a_x, a_y, a_z, phi, the, psi):
    g       = 9.8 # [m/s^2]
    cos_the = np.cos(the)
    phi     = np.arcsin(-a_y / (g * cos_the))
    the     = np.arcsin(a_x / g)
    psi     = psi

    return phi, the, psi

Nsamples = 41500
dt       = 0.01
time     = np.arange(Nsamples) * dt
phi_save = np.zeros(Nsamples)
the_save = np.zeros(Nsamples)
psi_save = np.zeros(Nsamples)

phi, the, psi = 0, 0, 0
for i in range(Nsamples):
    a_x, a_y, a_z = get_Accel(i)
    phi, the, psi = convert_Accel2Euler(a_x, a_y, a_z, phi, the, psi)
    phi_save[i]   = np.rad2deg(phi)
    the_save[i]   = np.rad2deg(the)
    psi_save[i]   = np.rad2deg(psi)
    
    
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 20))

plt.subplot(3, 1, 1)
plt.plot(time, phi_save, 'r', label='Roll ($\\phi$)', markersize=0.2)
plt.legend(loc='lower right')
plt.title('Roll ($\\phi$)')
plt.xlabel('Time [sec]')
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
    
    
    