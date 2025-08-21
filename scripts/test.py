import numpy as np

data = np.load(r"C:\Users\danie\OneDrive\Desktop\FlightPhase\FlightPhase\scripts\tfrrs_performances.npy", allow_pickle=True)
print(data.shape)
print(data[0])