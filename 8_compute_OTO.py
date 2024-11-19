#%%
import numpy as np
import h5py
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

def write_xyz(trajectory, analysis, outfile, comment_line):
    with open(outfile, "w") as file:
        for t in range(trajectory.shape[1]):
            file.write(f"{str(trajectory.shape[0])}\n")
            file.write(f"{comment_line}\n")
            for i in range(trajectory.shape[0]):
                file.write(f"{str(trajectory[i,t,0])} {str(trajectory[i,t,1])} {str(trajectory[i,t,2])} {str(analysis[i,t])}\n")

def compute_angle(v1, v2):
    dot_prod = np.dot(v1, v2)
    norme = np.linalg.norm(v1) * np.linalg.norm(v2)
    cos_angolo = dot_prod / norme
    return np.arccos(cos_angolo)

def compute_distance(p1, p2, Lx, Ly, Lz):
    dp = np.abs(p2-p1)
    dp = np.minimum(dp, np.array([Lx,Ly,Lz]) - dp)
    return np.sqrt(np.sum(dp**2))

def compute_distance_from_nth_atom(trajectory, atom,t, n, Lx, Ly, Lz):
    dp = np.abs(trajectory[:, t, :] - trajectory[atom, t, :])
    dp = np.minimum(dp, np.array([Lx, Ly, Lz]) - dp)
    dist_squared = np.sum(dp**2, axis=1)
    dist = np.sqrt(dist_squared)
    nth_distance = np.sort(dist)[n]
    index_1 = np.where(dist == 0.0)[0][0]
    index_2 = np.where(dist == nth_distance)[0][0]
    
    return nth_distance, index_1, index_2

def find_nth_neigh(trajectory, atom,t, n, Lx, Ly, Lz):
    dp = np.abs(trajectory[:, t, :] - trajectory[atom, t, :])
    dp = np.minimum(dp, np.array([Lx, Ly, Lz]) - dp)
    dist_squared = np.sum(dp**2, axis=1)
    dist = np.sqrt(dist_squared)
    n_neigh = np.sort(dist)[1:n+1]

    index_1 = np.zeros(n_neigh.shape)
    index_2 = np.zeros(n_neigh.shape)
    for i in range(n_neigh.shape[0]):
        index_1[i] = np.where(dist == 0.0)[0][0]
        index_2[i] = np.where(dist == n_neigh[i])[0][0]

    indexes = np.column_stack((index_1,index_2)).astype(int)
    return indexes

def update_coordinate(trajectory, neighbors ,atom,t, n, Lx, Ly, Lz):
    new_coordinate = np.zeros([neighbors.shape[0], 3, 2])
    for i in range(neighbors.shape[0]):
        comp_1 = trajectory[neighbors[i][0],t,:].copy()
        comp_2 = trajectory[neighbors[i][1],t,:].copy()
        dx = np.abs(comp_2[0]-comp_1[0])
        dy = np.abs(comp_2[1]-comp_1[1])
        dz = np.abs(comp_2[2]-comp_1[2])

        if dx > Lx / 2:
            if(comp_1[0] > Lx / 2):
                comp_2[0] += Lx
            else:
                comp_2[0] -= Lx
        if dy > Ly / 2:
            if(comp_1[1] > Ly / 2):
                comp_2[1] += Ly
            else:
                comp_2[1] -= Ly
        if dz > Lz / 2:
            if(comp_1[2] > Lz / 2):
                comp_2[2] += Lz
            else:
                comp_2[2] -= Lz

        new_coordinate[i,:,0] = comp_1
        new_coordinate[i,:,1] = comp_2
    return new_coordinate
    
def compute_oto(filename, box):
    trajectory = read_from_xyz(filename)
    results = np.zeros((trajectory.shape[0],trajectory.shape[1]))
    for t in range(trajectory.shape[1]):
        print(f"FRAME: {t}")
        for p in range(trajectory.shape[0]):
            neighbors = find_nth_neigh(trajectory,p,t,4, box[t,0], box[t,1], box[t,2])
            coordinate = update_coordinate(trajectory,neighbors,p,t,4, box[t,0], box[t,1], box[t,2])
            vectors = coordinate[:,:,1] - coordinate[0,:,0]
            vettori_norm = vectors / np.linalg.norm(vectors, axis=1)[:, np.newaxis]
            cos_phi = np.dot(vettori_norm, vettori_norm.T)
            cos_phi[np.diag_indices_from(cos_phi)] = 0
            q = 0
            n = len(vectors)
            for i in range(n):
                for j in range(i+1, n):
                    q += (cos_phi[i, j] + 1/3)**2

            q = 1 - (3/8) * q
            results[p,t] = q          
    return results
#DEBUG  
print(f"{'-'*10}\nOTO\n{'-'*10}")
traj_name = "ice_water_O"
in_file = "ice_water_O.hdf5"
with h5py.File(in_file, "r") as file:
    traj_array = np.array(file[f"Trajectories/{traj_name}/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file[f"Trajectories/{traj_name}/Box"])
print(box_array.shape)
q = compute_oto("trajectory.xyz", box_array)
# trajectory = read_from_xyz("try.xyz")
# write_xyz(trajectory,q,"oto.xyz", "Properties=pos:R:3:color:S:1")
np.save("arrays/OTO.npy",q)

# %% Spatial smoothing
input_file = "ice_water_O.hdf5"
with h5py.File(input_file, "r") as file:
    traj_array = np.array(file["Trajectories/ice_water_O/Trajectory"])
    traj_array = traj_array.transpose(1,0,2)
    box_array = np.array(file["Trajectories/ice_water_O/Box"])
sp_cutoff = [10]
for cutoff in sp_cutoff:
    input_array = f"arrays/OTO.npy"
    print(f"SPATIAL SMOOTHING {cutoff} - ({input_array})")
    res_array = f"arrays/sp_{cutoff}_OTO.npy"
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