#%%
import MDAnalysis as mda
import numpy as np
from MDAnalysis.analysis.distances import distance_array
from MDAnalysis.lib.distances import apply_PBC

# Load the universe
u = mda.Universe("simulation/ice_water.gro", "simulation/ice_water_500.xtc")

# Select oxygen atoms
oxygen_atoms = u.select_atoms("type O")
num_oxygens = len(oxygen_atoms)
num_frames = len(u.trajectory)

# Initialize the numpy array to store tetrahedral order parameters
tetrahedral_order = np.zeros((num_oxygens, num_frames))

# Function to apply minimum image convention
def minimum_image(vector, box):
    for i in range(3):
        if vector[i] > 0.5 * box[i]:
            vector[i] -= box[i]
        elif vector[i] < -0.5 * box[i]:
            vector[i] += box[i]
    return vector

# Function to compute the tetrahedral order parameter for a given frame
def compute_tetrahedral_order(atoms, frame_index, box):
    positions = atoms.positions
    distances = distance_array(positions, positions, box=box)
    np.fill_diagonal(distances, np.inf)  # Ignore self-distances

    for i, pos in enumerate(positions):
        # Find the four nearest neighbors
        nearest_neighbors = np.argsort(distances[i])[:4]
        angles = []

        # Calculate the angles between pairs of nearest neighbors
        for j in range(len(nearest_neighbors)):
            for k in range(j + 1, len(nearest_neighbors)):
                neighbor1_pos = positions[nearest_neighbors[j]]
                neighbor2_pos = positions[nearest_neighbors[k]]

                vec1 = neighbor1_pos - pos
                vec2 = neighbor2_pos - pos
                vec1 = minimum_image(vec1, box)
                vec2 = minimum_image(vec2, box)

                cos_theta = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                #theta = np.arccos(np.clip(cos_theta, -1.0, 1.0))  # Ensure within valid range for arccos
                #angles.append(theta)
                print(cos_theta)
                angles.append(cos_theta)
        angles = np.array(angles)
        #print(angles.shape)

        # Calculate the tetrahedral order parameter for this atom
        q_tet = 1 - (3 / 8) * np.sum((angles + 1/3)**2)
        tetrahedral_order[i, frame_index] = q_tet

# Iterate over all frames and compute the tetrahedral order parameter for each frame
#box = [32.0,30.0,68.28799,90.0,90.0,90.0]  # Get the periodic box dimensions
for frame_index, ts in enumerate(u.trajectory):
    box = u.dimensions
    print(box)
    compute_tetrahedral_order(oxygen_atoms, frame_index, box)

# Output the results
print("Tetrahedral order parameter array shape:", tetrahedral_order.shape)
print("Tetrahedral order parameter for the system:\n", tetrahedral_order)
np.save("arrays/try_oto.npy",tetrahedral_order)
# %%
arra = np.load("arrays/try_oto.npy")
for i in range(arra.shape[0]):
    for j in range(arra.shape[1]):
        if arra[i,j]< 0:
            print(i)
            print(j)
            print(f"WARNING: {arra[i,j]}")
# %%
