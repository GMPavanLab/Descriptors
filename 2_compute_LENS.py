#THIS CODE IS USED TO COMPUTE LENS DESCRIPTOR
import h5py
import numpy as np
import dynsight 
#LENS cutoff
LENS_CUTOFF = 10
#Load HDF5 file (see code 1_*)
in_file = "ice_water_O.hdf5"
traj_name = "ice_water_O"
frames_range = slice(0,500)

print(f"{'-'*10}\nLENS\n{'-'*10}")
with h5py.File(in_file,"r") as file:
    tgroup = file["Trajectories"][traj_name]
    print(tgroup)
    universe = dynsight.hdf5er.create_universe_from_slice(tgroup, frames_range)
print("Computing LENS")
neig_counts = dynsight.lens.list_neighbours_along_trajectory(universe, cutoff = LENS_CUTOFF)    
LENS, nn, *_=dynsight.lens.neighbour_change_in_time(neig_counts)
results = np.array([LENS,nn])   
print("Saving LENS results")
with h5py.File(in_file, "r+") as file:
    file["LENS"].create_dataset(f"LENS_{int(LENS_CUTOFF)}", data=results)
with h5py.File(in_file, "r") as file:
    LENS = np.array(file["LENS"][f"LENS_{int(LENS_CUTOFF)}"][0,:,:])
    np.save(f"arrays/LENS_{int(LENS_CUTOFF)}",LENS)

# Local denoising (Spatial smoothing)
input_file = "ice_water_O.hdf5"
with h5py.File(input_file, "r") as file:
    traj_array = np.array(file["Trajectories/ice_water_O/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file["Trajectories/ice_water_O/Box"])
sp_cutoff = [10]
for cutoff in sp_cutoff:
    input_array = f"arrays/LENS_{int(LENS_CUTOFF)}.npy"
    print(f"SPATIAL SMOOTHING {cutoff} - ({input_array})")
    res_array = f"arrays/sp_{cutoff}_LENS_{int(LENS_CUTOFF)}.npy"
    volume_shape = "sphere"
    descriptor = np.load(input_array)
    print(descriptor.shape)
    descriptor = descriptor.T
    averaged = dynsight.data_processing.spatialaverage(traj_array,
                                                    box_array,
                                                    descriptor,
                                                    cutoff=cutoff, 
                                                    volume_shape = volume_shape)
    np.save(res_array,averaged.T)
