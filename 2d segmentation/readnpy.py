import numpy as np
data = np.load('outputs/rendered_716/mask_data/mask_0001.npy')
print(data.shape)
np.savetxt('data.txt', data)