import numpy as np
import os
ind="1"
path=r"C:\Users\15512\Desktop\OverlapPredator\labels" +f"\{ind}" + ".npy"
print(path)  
test=np.load(path,allow_pickle=True)
print(type(test))
print(np.max(test))
print(test.shape)


