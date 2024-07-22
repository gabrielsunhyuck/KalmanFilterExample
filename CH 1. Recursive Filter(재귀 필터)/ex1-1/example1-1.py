####### 평균 필터 계산 ########

import numpy as np
import random
import matplotlib.pyplot as plt

class averageFilter():
    def __init__(self):
        self.avg_prev = 0
        self.k = 0
        
    def getVolt(self):
        self.w = 0 + 4*random.random()
        self.z = 14.4 + self.w
        return self.z

    def avgFilter(self, x):
        self.k += 1
        self.alpha = (self.k - 1)/self.k
        self.avg = self.alpha * self.avg_prev + (1 - self.alpha)*x
        self.avg_prev = self.avg
        
        return self.avg
    
    def plotting(self):
        t = np.arange(0,10,0.2)
        Nsamples = len(t)

        xm = np.zeros(Nsamples)
        volt_average = np.zeros(Nsamples)

        for i in range(Nsamples):
            x = averageFilter.getVolt(self)
            avg = averageFilter.avgFilter(self,x)

            xm[i] = x
            volt_average[i] = avg

        plt.plot(t, xm, 'b*--', label='Measured')
        plt.plot(t, volt_average, 'ro', label='Average')
        plt.legend(loc='upper left')
        plt.ylabel('Volt [V]')
        plt.xlabel('Time [sec]')
        plt.show()
        
filter_instance = averageFilter()
filter_instance.plotting()