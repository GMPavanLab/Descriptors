#%%
import numpy as np
import os 

directory = "arrays"
descriptors = []
for filename in os.listdir(directory):
    if(filename.endswith("SOAP_10.npy")):
        continue
    if os.path.isfile(os.path.join(directory, filename)):
        descriptors.append(filename)

print(descriptors)
print(f"{'-'*10}\nDESCRIPTOR CHECKS\n{'-'*10}")
for d in descriptors:
    arr = np.load(f"arrays/{d}")
    print(f"descriptor:     {d[:-4].ljust(20)} -      shape: {arr.shape}")
# %%
