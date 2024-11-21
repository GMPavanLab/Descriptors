#THIS CODE IS USED TO COMPUTE d5
import h5py
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import distance_array
import dynsight

print(f"{'-'*10}\nDIST 5\n{'-'*10}")
simulation_folder = "simulation"
topo_file = f"{simulation_folder}/ice_water.gro"
traj_file = f"{simulation_folder}/ice_water_500.xtc"
u = mda.Universe(topo_file, traj_file)
selection = u.select_atoms("type O")
a = 0
dist_5 = np.zeros((2048,500))
for ts in u.trajectory:
    distances = distance_array(selection.positions, selection.positions, box=u.dimensions)
    sort_distances = np.sort(distances)
    id = np.argsort(distances)
    for i in range(0,6):
        print(f"atom {i}) coord: {selection.positions[i]}")
    for i in range(sort_distances.shape[0]):
        dist_5[i,a] = sort_distances[i,5]
    a += 1
    dist_5 = np.array(dist_5)
np.save("arrays/dist_5.npy",dist_5)
# Local denoising (Spatial smoothing)
input_file = "ice_water_O.hdf5"
with h5py.File(input_file, "r") as file:
    traj_array = np.array(file["Trajectories/ice_water_O/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file["Trajectories/ice_water_O/Box"])
sp_cutoff = [10]
for cutoff in sp_cutoff:
    input_array = f"arrays/dist_5.npy"
    print(f"SPATIAL SMOOTHING {cutoff} - ({input_array})")
    res_array = f"arrays/sp_{cutoff}_dist_5.npy"
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