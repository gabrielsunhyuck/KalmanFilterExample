######### 영상 속의 물체 추적하기 예제 #########

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.metrics import structural_similarity

np.random.seed(0)

def get_ball_centerPosition(iimg=0):
 
    imageA = cv2.imread('/home/sun/KalmanFilter/CH10.tracking object in video/10.TrackKalman/Img/bg.jpg')
    imageB = cv2.imread('/home/sun/KalmanFilter/CH10.tracking object in video/10.TrackKalman/Img/{}.jpg'.format(iimg+1))

    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    _, diff = structural_similarity(grayA, grayB, full=True)
    diff = (diff * 255).astype('uint8') 

    thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    M = cv2.moments(contours[0])
    x_center = int(M['m10'] / M['m00'])  
    y_center = int(M['m01'] / M['m00']) 
    print("x, y :", x_center, y_center)

    v_k = np.random.normal(0, 15) 

    z_x = x_center + v_k   
    z_y = y_center + v_k   
    print("z_x, z_y :",z_x, z_y)
    
    return np.array([z_x, z_y])

def KalmanFilter(x_esti, z_k, P, Q, R):
    # (1) 추정값과 오차 공분산 예측
    x_pred = A @ x_esti
    P_pred = A @ P @ A.T + Q
    # (2) 칼만 이득 계산
    K = P_pred @ H.T @ np.linalg.inv(H @ P_pred @ H.T + R)
    # (3) 추정값 계산
    x_esti = x_pred + K @ (z_k - H @ x_pred)
    # (4) 오차 공분산 계산
    P = P_pred - K @ H @ P_pred
    
    return x_esti, P, K

dt       = 1
Nsamples = 24

x_position_measured1 = np.zeros(Nsamples)
x_position_estimate1 = np.zeros(Nsamples)
y_position_measured1 = np.zeros(Nsamples)
y_position_estimate1 = np.zeros(Nsamples)

x_position_measured2 = np.zeros(Nsamples)
x_position_estimate2 = np.zeros(Nsamples)
y_position_measured2 = np.zeros(Nsamples)
y_position_estimate2 = np.zeros(Nsamples)

x_position_measured3 = np.zeros(Nsamples)
x_position_estimate3 = np.zeros(Nsamples)
y_position_measured3 = np.zeros(Nsamples)
y_position_estimate3 = np.zeros(Nsamples)

x_position_measured4 = np.zeros(Nsamples)
x_position_estimate4 = np.zeros(Nsamples)
y_position_measured4 = np.zeros(Nsamples)
y_position_estimate4 = np.zeros(Nsamples)

A = np.array([[1, dt, 0, 0],
              [0, 1, 0, 0],
              [0, 0, 1, dt],
              [0, 0, 0, 1]])
H = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])
Q1 = np.eye(4)
Q2 = (1/100)*np.eye(4)
R = np.array([[50, 0],
              [0, 50]])
R2= 50*np.array([[50, 0],
                  [0, 50]])
R3= (1/50)*np.array([[50, 0],
                  [0, 50]])

x_0 = np.array([0, 0, 0, 0])  
P_0 = 100 * np.eye(4)

x_esti, P = None, None
for i in range(Nsamples):
    z_k = get_ball_centerPosition(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = KalmanFilter(x_esti, z_k, P, Q1, R)
        
    x_position_measured1[i] = z_k[0]
    x_position_estimate1[i] = x_esti[0]
    y_position_measured1[i] = z_k[1]
    y_position_estimate1[i] = x_esti[2]

for i in range(Nsamples):
    z_k = get_ball_centerPosition(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = KalmanFilter(x_esti, z_k, P, Q2, R)
        
    x_position_measured2[i] = z_k[0]
    x_position_estimate2[i] = x_esti[0]
    y_position_measured2[i] = z_k[1]
    y_position_estimate2[i] = x_esti[2]
    
for i in range(Nsamples):
    z_k = get_ball_centerPosition(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = KalmanFilter(x_esti, z_k, P, Q1, R2)
        
    x_position_measured3[i] = z_k[0]
    x_position_estimate3[i] = x_esti[0]
    y_position_measured3[i] = z_k[1]
    y_position_estimate3[i] = x_esti[2]
    
for i in range(Nsamples):
    z_k = get_ball_centerPosition(i)
    if i == 0:
        x_esti, P = x_0, P_0
    else:
        x_esti, P, K = KalmanFilter(x_esti, z_k, P, Q1, R3)
        
    x_position_measured4[i] = z_k[0]
    x_position_estimate4[i] = x_esti[0]
    y_position_measured4[i] = z_k[1]
    y_position_estimate4[i] = x_esti[2]
    
fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis() 
plt.plot(x_position_measured1, y_position_measured1, 'ko', markersize=5, label='Measured Point')
plt.plot(x_position_estimate1, y_position_estimate1, 'r*', markersize=8, label='Estimated Point')
plt.xlabel('x [pixel]'); plt.ylabel('y [pixel]')
plt.title('Position : Measurement vs. Estimation')
plt.legend(loc='lower right')
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis()
plt.plot(x_position_estimate2, y_position_estimate2, 'ko', markersize=5, label='1/100 * Q')
plt.plot(x_position_estimate1, y_position_estimate1, 'r*', markersize=8, label='Q')
plt.xlabel('x [pixel]'); plt.ylabel('y [pixel]')
plt.title('Position : Q vs. 1/100*Q')
plt.legend(loc='lower right')
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis()
plt.plot(x_position_estimate3, y_position_estimate3, 'ko', markersize=5, label='50 * R')
plt.plot(x_position_estimate1, y_position_estimate1, 'r*', markersize=8, label='R')
plt.xlabel('x [pixel]'); plt.ylabel('y [pixel]')
plt.title('Position : R vs. 50*R')
plt.legend(loc='lower right')
plt.show()

fig = plt.figure(figsize=(8, 8))
plt.gca().invert_yaxis()
plt.plot(x_position_estimate3, y_position_estimate3, 'ko', markersize=5, label='(1/50) * R')
plt.plot(x_position_estimate1, y_position_estimate1, 'r*', markersize=8, label='R')
plt.xlabel('x [pixel]'); plt.ylabel('y [pixel]')
plt.title('Position : R vs. (1/50)*R')
plt.legend(loc='lower right')
plt.show()
