import numpy as np

magnitude = np.random.rand(512, 32, 32)
orientation_bins = np.zeros((512, 32, 32, 9))

orientation_bins[:, :, :, 0] = np.where(magnitude < 1, magnitude, 0)