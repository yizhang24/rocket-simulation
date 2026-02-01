import re
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import fft
from scipy.signal import butter, detrend, filtfilt, find_peaks, windows

assert len(sys.argv) == 2, "Wrong number of args, provide OpenRocket export"

# Import OpenRocket data and cleanse column names
df: pd.DataFrame = pd.read_csv(sys.argv[1], comment="#")
df = df.rename(columns=lambda column_name: re.sub(r'\([^)]*\)', '', column_name).strip())

# Crop to descent phase (apogee -> landing)
apogee_idx = df['Altitude'].idxmax()
landing_idx = df['Altitude'].index[-1]
df_cropped = df.loc[apogee_idx:landing_idx].copy()
print('Selecting descent data...')
print(f'  Apogee time: {df_cropped['Time'].iloc[0]} sec')
print(f'  Apogee altitude: {df_cropped['Altitude'].iloc[0]} meters')
print(f'  Landing altitude: {df_cropped['Altitude'].iloc[-1]} meters')
print(f'  Descent duration: {df_cropped['Time'].iloc[-1] - df_cropped['Time'].iloc[0]} sec\n')

# Get data series
time = df_cropped['Time'].to_numpy()

vertical_accel = df_cropped['Vertical acceleration'].to_numpy()
lateral_accel = df_cropped['Lateral acceleration'].to_numpy()
total_accel = df_cropped['Total acceleration'].to_numpy()
pos_x = df_cropped['Position East of launch'].to_numpy()
pos_y = df_cropped['Position North of launch'].to_numpy()
altitude = df_cropped['Altitude'].to_numpy()

def frequency_analysis(time: np.ndarray, acceleration: np.ndarray):
  dt = np.mean(np.diff(time))

  accel_processed = detrend(acceleration)

  window = windows.hann(len(accel_processed))
  accel_processed = accel_processed * window

  n = len(accel_processed)
  fft_vals = fft.fft(accel_processed) # Array of vals
  fft_freq = fft.fftfreq(n, dt) # Bins
  pos_mask = fft_freq > 0
  frequencies = fft_freq[pos_mask]
  amplitudes = 2 * np.abs(fft_vals[pos_mask]) / n

  psd = amplitudes ** 2
  peaks, props = find_peaks(psd, height=np.max(psd)/10, distance=10)
  dominant_freqs = frequencies[peaks]
  dominant_amps = amplitudes[peaks]
  return frequencies, amplitudes, psd, dominant_freqs, dominant_amps

frequencies, amplitudes, psd, dominant_freqs, dominant_amps = frequency_analysis(time, lateral_accel)

# Cool plots
N_ROWS = 2
N_COLS = 2
fig = plt.figure(figsize=(20, 14))
plot_id = 1

ax: plt.Axes = fig.add_subplot(N_ROWS, N_COLS, plot_id, projection='3d')
ax.scatter(pos_x, pos_y, altitude, c = plt.cm.jet(time/max(time)))
ax.set_xlabel('$X$')
ax.set_ylabel('$Y$')
ax.set_zlabel('$Z$')
plot_id+=1

ax: plt.Axes = fig.add_subplot(N_ROWS, N_COLS, plot_id)
ax.plot(frequencies, amplitudes)
plot_id+=1

ax: plt.Axes = fig.add_subplot(N_ROWS, N_COLS, plot_id)
ax.plot(time, vertical_accel)
plot_id+=1

ax: plt.Axes = fig.add_subplot(N_ROWS, N_COLS, plot_id)
ax.plot(time, altitude)
plot_id+=1

plt.show()
