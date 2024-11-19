#%%
import h5py
import numpy as np
import dynsight
#%%
def read_from_xyz(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
        n_parts_str = lines[0]
        frames = 0
        for i in range(len(lines)):
            if(lines[i] == n_parts_str):
                frames += 1
                continue
            if(lines[i-1] == n_parts_str):
                continue

        n_parts = int(n_parts_str)
        trajectory = np.zeros((n_parts, frames, 3))
        time = -1
        atom = 0
        for i in range(len(lines)):
            if(lines[i] == n_parts_str):
                time += 1
                atom = 0
                continue
            if(lines[i-1] == n_parts_str):
                atom = 0
                continue

            comp = lines[i].split()

            trajectory[atom, time, 0] = float(comp[1])
            trajectory[atom, time, 1] = float(comp[2])
            trajectory[atom, time, 2] = float(comp[3])
            atom += 1
        return trajectory
    
def compute_distance(p1, p2, Lx, Ly, Lz):
    dp = np.abs(p2-p1)
    dp = np.minimum(dp, np.array([Lx,Ly,Lz]) - dp)
    return np.sqrt(np.sum(dp**2))

def compute_displacement(absolute,box):
    results = np.zeros((trajectory.shape[0],trajectory.shape[1]))
    for t in range(trajectory.shape[1]):
        if(t==0):
            results[:,t] = 0.0
            continue
        #print(f"FRAME: {t}")
        for p in range(trajectory.shape[0]):
            if(absolute):
                results[p,t] = compute_distance(trajectory[p,0],trajectory[p,t], box[t,0], box[t,1], box[t,2])
            else:
                results[p,t] = compute_distance(trajectory[p,t],trajectory[p,t-1], box[t,0], box[t,1], box[t,2])
    return results

print(f"{'-'*10}\nDISP\n{'-'*10}")
in_file = "ice_water_O.hdf5"
traj_name = "ice_water_O"

traj = read_from_xyz("trajectory.xyz")

with h5py.File(in_file, "r") as file:
    traj_array = np.array(file[f"Trajectories/{traj_name}/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file[f"Trajectories/{traj_name}/Box"])
trajectory = traj_array
#disp_abs = compute_displacement(True)
disp_rel = compute_displacement(False,box_array)
disp_rel = disp_rel / 100 
np.save("arrays/vel.npy", disp_rel)
#np.save("arrays/disp_abs.npy", disp_abs)
# %% Spatial smoothing
input_file = "ice_water_O.hdf5"
with h5py.File(input_file, "r") as file:
    traj_array = np.array(file["Trajectories/ice_water_O/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file["Trajectories/ice_water_O/Box"])
sp_cutoff = [10]
for cutoff in sp_cutoff:
    input_array = f"arrays/vel.npy"
    print(f"SPATIAL SMOOTHING {cutoff} - ({input_array})")
    res_array = f"arrays/sp_{cutoff}_vel.npy"
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