####### 저주파 통과 필터 ########

import numpy as np
import matplotlib.pyplot as plt
from scipy import io

class LPF():
    def __init__(self):
        self.n         = 10
        self.Nsamples  = 500
        self.end_Time  = 10
        self.dt        = 0.02
        self.first     = True
        self.time      = np.arange(0, self.end_Time, self.dt)
        self.x_measure = np.zeros(self.Nsamples)
        self.x_average = np.zeros(self.Nsamples)
                
    def sonar(self, i):
        input_sonar = io.loadmat('/home/sun/KalmanFilter/source/2.MovAvgFilter/SonarAlt.mat')
        self.sonar_dist = input_sonar['sonarAlt'][0][i]
        return self.sonar_dist

    def LPF(self, x, alpha):
        if self.first:
            self.x_prev = x
            self.first = False
        self.x_LPF  = alpha * self.x_prev + (1 - alpha) * x
        self.x_prev = self.x_LPF
        return self.x_LPF
        
    def plotting(self):
        for i in range(0, self.Nsamples):
            xm     = LPF.sonar(self, i)
            self.x = LPF.LPF(self, xm, 0.5) ## [ 0 < alpha < 1 ]
            
            self.x_measure[i] = xm
            self.x_average[i] = self.x
                    
        plt.scatter(self.time, self.x_measure, label='Measured', s=20, c = 'k', marker='o', alpha=0.5)
        plt.plot(self.time, self.x_average, 'r', label='LPF(alpha = 0.5)') 
        plt.legend(loc='upper left')
        plt.xlabel('time [s]')
        plt.ylabel('altitude [m]')
        plt.show()
        
filter_instance = LPF()
filter_instance.plotting()
            