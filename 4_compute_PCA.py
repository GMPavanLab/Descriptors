#%%
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import pandas as pd
SOAP_CUTOFF = 10
pc_components = 3
#%%
print(f"{'-'*10}\nPCA SOAP\n{'-'*10}")
soap_av = np.load(f"arrays/fullvect_SOAP_{SOAP_CUTOFF}.npy")
print(soap_av.shape)
SOAP=soap_av.reshape(np.shape(soap_av)[0]*np.shape(soap_av)[1],np.shape(soap_av)[2])

pca = PCA(n_components=pc_components)
pc_soap = pca.fit_transform(SOAP)
np.save(f"arrays/SOAP_{SOAP_CUTOFF}_PC1.npy",np.transpose(pc_soap[:,0].reshape(500,2048)))
#np.save(f"arrays/SOAP_{SOAP_CUTOFF}_PC2.npy",np.transpose(pc_soap[:,1].reshape(500,2048)))
#np.save(f"arrays/SOAP_{SOAP_CUTOFF}_PC3.npy",np.transpose(pc_soap[:,2].reshape(500,2048)))
# %%
print(f"{'-'*10}\nPCA SP SOAP\n{'-'*10}")
sp_cutoff = [10]
for cutoff in sp_cutoff:
    soap_av = np.load(f"arrays/sp_{cutoff}_SOAP_{int(SOAP_CUTOFF)}.npy")
    soap_av = soap_av.transpose(2,1,0)
    print(soap_av.shape)
    SOAP=soap_av.reshape(np.shape(soap_av)[0]*np.shape(soap_av)[1],np.shape(soap_av)[2])

    pca = PCA(n_components=pc_components)
    pc_soap = pca.fit_transform(SOAP)
    np.save(f"arrays/sp_{cutoff}_SOAP_{SOAP_CUTOFF}_PC1.npy",np.transpose(pc_soap[:,0].reshape(500,2048)))
    #np.save(f"arrays/sp_{cutoff}_SOAP_{SOAP_CUTOFF}_PC2.npy",np.transpose(pc_soap[:,1].reshape(500,2048)))
    #np.save(f"arrays/sp_{cutoff}_SOAP_{SOAP_CUTOFF}_PC3.npy",np.transpose(pc_soap[:,2].reshape(500,2048)))
