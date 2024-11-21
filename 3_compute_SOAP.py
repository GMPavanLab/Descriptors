#THIS CODE IS USED TO COMPUTE SOAP DESCRIPTOR
import h5py
import dynsight
import numpy as np
#SOAP cutoff
SOAP_CUTOFF = 10

traj_name = "ice_water_O"
in_file = f"{traj_name}.hdf5"

print(f"{'-'*10}\nSOAP\n{'-'*10}")
print("Computing SOAP")
with h5py.File(in_file, "r+") as file:
    dynsight.soapify.saponify_trajectory(
        trajcontainer= file["Trajectories"][traj_name],
        soapoutcontainer= file["SOAP"].require_group(f"SOAP_{int(SOAP_CUTOFF)}"),
        verbose= False,
        soapnmax=8,
        soaplmax= 8,
        soapnjobs = 8,
        soaprcut= SOAP_CUTOFF
    )
print("Saving SOAP results")
with h5py.File(in_file, "r") as file:
    soap = np.array(file["SOAP"][f"SOAP_{int(SOAP_CUTOFF)}"][traj_name])
filled_soap = dynsight.soapify.fill_soap_vector_from_dscribe(soap,lmax=8,nmax=8)
with h5py.File(in_file, "a") as file:
    file["SOAP"][f"SOAP_{int(SOAP_CUTOFF)}"].create_dataset("fill_SOAP", data=filled_soap)
np.save(f"arrays/fullvect_SOAP_{int(SOAP_CUTOFF)}", filled_soap)

# Local denoising (Spatial smoothing)
input_file = "ice_water_O.hdf5"
with h5py.File(input_file, "r") as file:
    traj_array = np.array(file["Trajectories/ice_water_O/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file["Trajectories/ice_water_O/Box"])
sp_cutoff = [10]
for cutoff in sp_cutoff:
    input_array = f"arrays/fullvect_SOAP_{int(SOAP_CUTOFF)}.npy"
    print(f"SPATIAL SMOOTHING {cutoff} - ({input_array})")
    res_array = f"arrays/sp_{cutoff}_SOAP_{int(SOAP_CUTOFF)}.npy"
    volume_shape = "sphere"
    descriptor = np.load(input_array)
    descriptor = np.transpose(descriptor,(2,1,0))
    print(descriptor.shape)
    descriptor = descriptor.T
    averaged = dynsight.data_processing.spatialaverage(traj_array,
                                                    box_array,
                                                    descriptor,
                                                    cutoff=cutoff, 
                                                    volume_shape = volume_shape)
    np.save(res_array,averaged.T)
