#%% 
import h5py
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import distance_array
import dynsight

nn_cutoff = 10
#%%
def compute_distance(p1, p2, Lx, Ly, Lz):
    dp = np.abs(p2-p1)
    dp = np.minimum(dp, np.array([Lx,Ly,Lz]) - dp)
    return np.sqrt(np.sum(dp**2))
print(f"{'-'*10}\nNN\n{'-'*10}")
simulation_folder = "simulation"
topo_file = f"{simulation_folder}/ice_water.gro"
traj_file = f"{simulation_folder}/ice_water_500.xtc"
u = mda.Universe(topo_file, traj_file)
#selected_atoms = u.select_atoms("type 1 or type 2")
print(u.dimensions)


selection = u.select_atoms("type O or type H")

coord_numbers_per_frame = []
for ts in u.trajectory:
    distances = distance_array(selection.positions, selection.positions, box=u.dimensions)
    coordination_number = (distances < nn_cutoff).sum(axis=1) - 1
    print(f'Frame {ts.frame}: {coordination_number}')
    coord_numbers_per_frame.append(coordination_number)
nn = np.array(coord_numbers_per_frame)
nn = nn.T
np.save(f"arrays/nn_{nn_cutoff}.npy", nn[::3,:])

# %% Spatial smoothing
input_file = "ice_water_O.hdf5"
with h5py.File(input_file, "r") as file:
    traj_array = np.array(file["Trajectories/ice_water_O/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file["Trajectories/ice_water_O/Box"])
sp_cutoff = [10]
for cutoff in sp_cutoff:
    input_array = f"arrays/nn_{nn_cutoff}.npy"
    print(f"SPATIAL SMOOTHING {cutoff} - ({input_array})")
    res_array = f"arrays/sp_{cutoff}_nn_{nn_cutoff}.npy"
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