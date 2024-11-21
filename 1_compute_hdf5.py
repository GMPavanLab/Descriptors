#THIS CODE IS USED TO CONVERT .GRO AND .XTC TRAJECTORY INTO
#HDF5 DATABASE
#.GRO AND .XTC ZENODO LINK CAN BE FOUND IN THE ARTICLE

#Libraries
import MDAnalysis as mda
import dynsight 
import h5py
import numpy as np
# HDF5 file build
print(f"{'-'*10}\nSTART INITILIAZIATION PART\n{'-'*10}")
traj_name = "ice_water_O"
simulation_folder = "simulation"
topo_file = f"{simulation_folder}/ice_water.gro"
traj_file = f"{simulation_folder}/ice_water.xtc"
out_file = f"{traj_name}.hdf5"
#Trajectory 50 ns 100 ps steps
frames_range = slice(0,12500,25)
print("Building HDF5 file:\n")
u = mda.Universe(topo_file, traj_file)
with mda.Writer(f"{simulation_folder}/ice_water_500.xtc", u.atoms.n_atoms) as W:
    for ts in u.trajectory[frames_range]:
        W.write(u)
traj_file = f"{simulation_folder}/ice_water_500.xtc"
u = mda.Universe(topo_file,traj_file)
dynsight.hdf5er.mda_to_hdf5(u, out_file, "ice_water")
in_file = out_file
#O atoms will be used as reference
with h5py.File(in_file,"r") as file:
    dataset_box = file["Trajectories"]["ice_water"]["Box"]
    dataset_traj = file["Trajectories"]["ice_water"]["Trajectory"]
    dataset_types = file["Trajectories"]["ice_water"]["Types"]
    box = np.array(dataset_box)
    traj = np.array(dataset_traj)
    types = np.array(dataset_types)
Ox_index = np.where(types == b"O")[0]
Ox_traj = traj[:,Ox_index,:]
Ox_types = types[Ox_index]
with h5py.File(in_file,"a") as file:
    file["Trajectories"].require_group(traj_name)
    file["Trajectories"][traj_name].create_dataset("Box", data=box)
    file["Trajectories"][traj_name].create_dataset("Trajectory", data = Ox_traj, chunks=(100,len(Ox_types),3))
    file["Trajectories"][traj_name].create_dataset("Types", data = Ox_types)
    #Prepare for descriptors computation...
    file.create_group("LENS")
    file.create_group("SOAP")
steps = Ox_traj.shape[0]
atoms = Ox_traj.shape[1]
#Just check
print("\nReading HDF5 file:\n")
print(f"Number of steps: {steps}\nNumber of particles: {atoms}\n")
