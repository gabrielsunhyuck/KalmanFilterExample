######### 칼만 필터 예제 8-1 #########

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

def getVolt():
    V         = np.random.normal(0,2)
    t_volt    = 14.4
    volt_meas = t_volt + V
    return volt_meas

def kalmanFilter(z, x_hat, P):
    global K
    ## (1) 추정값 예측
    x_bar = A * x_hat  # x_bar : 예측값, x_hat : 추정값
    ## (2) 오차 공분산 예측
    P_bar = A * P * A + Q
    ## (3) 칼만 이득 계산
    K     = (P_bar * H)/(H * P_bar * H + R)
    ## (4) 추정값 계산
    x_hat = x_bar + K * (z - H * x_bar)
    ## (5) 오차 공분산 계산
    P     = P_bar - K * H * P_bar
    return x_hat, P, K

A = 1; H = 1; Q = 0; R = 4
endTime = 10; dt = 0.2
x_0     = 14  # 초기 예측값
P_0     = 6   # 초기 예측 오차 공분산

t              = np.arange(0, endTime, dt)
Nsamples       = len(t)
volt_measured  = np.zeros(Nsamples)
volt_estimated = np.zeros(Nsamples)
P_calculated   = np.zeros(Nsamples)
K_changing     = np.zeros(Nsamples)
x_hat, P, K = None, None, None ## 초기 추정값과 오차 공분산은 알 수 없음(Null 값)

for i in range(Nsamples):
    z = getVolt()
    if i == 0:
        x_hat, P = x_0, P_0
    else:
        x_hat, P, K = kalmanFilter(z, x_hat, P)

    volt_measured[i]  = z
    volt_estimated[i] = x_hat
    P_calculated[i]   = P
    K_changing[i]     = K

plt.plot(t, volt_measured, 'r*--', label='Measurements')
plt.plot(t, volt_estimated, 'bo--', label='Estimations')
plt.plot(t, P_calculated, 'g:', label='Error Covirance')
plt.plot(t, K_changing, 'b:', label="Kalman gain")
plt.legend(loc='center right')
plt.title('Measurements vs. Estimations [K.F]')
plt.xlabel('time [s]')
plt.ylabel('voltage [V]')
plt.show()
plt.savefig('simple_kalman_filter.png')


