"""
python3 smooth_temperature.py sysinfo.csv
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pykalman import KalmanFilter

"""
LOESS Smoothing:
Adjust the frac parameter to get as much signal as possible with as little noise as possible. The contrasting factors: (1) when
the temperature spikes (because of momentary CPU usage), the high temperature values are reality and we don't want to
smooth that information out of existence, but (2) when the temperature is relatively flat (where the computer is not in use),
the temperature is probably relatively steady, not jumping randomly between 30°C and 33°C as the data implies
"""
cpu_data = pd.read_csv('sysinfo.csv', header=0)

lowess = sm.nonparametric.lowess

# Convert string timestamp to numerical
cpu_data['timestamp_f'] = pd.to_datetime(cpu_data['timestamp'])

x = cpu_data['timestamp_f']
y = cpu_data['temperature']
y_lws = lowess(y, x, frac=0.02, it=5)

plt.figure(figsize=(12,4))
plt.scatter(x, y, s=1, alpha=0.3, label="data")
plt.plot(x, y_lws[:, 1], 'r-', label="LOESS Smoothed")
#plt.show()

"""
Kalman Smoothing
A Kalman filter will let us take more information into account: we can use the processor usage, system load, and fan speed to
give a hint about when the temperature will be increasing/decreasing.

obs. covariance: how much you believe the sensors, error expected.
trans. covariance: how accurate your prediction is.
transition matrix: knowledge about the process; predict what the next vals of the variables observed will be.
    >> temperature, cpu_percent, sys_load_1, fan_rpm
"""

kalman_data = cpu_data[['temperature', 'cpu_percent', 'sys_load_1', 'fan_rpm']]
temperature_std = kalman_data['temperature'].std()
cpu_percent_std = kalman_data['cpu_percent'].std()
sys_load_1_std = kalman_data['sys_load_1'].std()
fan_rpm_std = kalman_data['fan_rpm'].std()

initial_state = kalman_data.iloc[0]
observation_covariance = np.diag([temperature_std,
                                  cpu_percent_std,
                                  sys_load_1_std,
                                  fan_rpm_std]) ** 2

transition_covariance = np.diag([temperature_std * 0.1,
                                 cpu_percent_std * 0.02,
                                 sys_load_1_std * 0.05,
                                 fan_rpm_std * .4]) ** 2

            #temp. #cpu_% #sys_load_1 #fan_rpm
transition = [[0.97, 0.5,   0.2, -0.001],     # temperature
              [0.1,  0.4,   2.2,    0],     # cpu_percent
              [0,    0,     0.95,   0],     # sys_load_1
              [0,    0,     0,      1]]     # fan_rpm


kf = KalmanFilter(
    initial_state_mean=initial_state,
    initial_state_covariance= observation_covariance,
    observation_covariance=observation_covariance,
    transition_covariance=transition_covariance,
    transition_matrices=transition
)
kalman_smoothed, _ = kf.smooth(kalman_data)
plt.plot(x, kalman_smoothed[:,0], 'g-', label="Kalman Smoothed")
plt.title("CPU Temperature Noise Reduction")
plt.xlabel("Date Time")
plt.ylabel("Temperature")
plt.legend()
plt.savefig('cpu.svg') # for final submission
plt.show()