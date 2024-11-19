#%%
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
fontsize = 18
lsize = 16
def onion_plot(plot_position, win,env0,states,log):
    width = 1
    win = win * 0.1
    fig,axes = plt.subplots(figsize=(5.5, 4.5),dpi=600)
    axes.plot(win, states, marker="o")
    axes.set_xlabel(r"Time resolution $\Delta t$ " + "[ns]",fontsize=fontsize)
    axes.set_ylabel(r"# of ENVs", weight="bold", c="#1f77b4",fontsize=fontsize)
    axes.tick_params(axis="y",colors="#1f77b4",width=width,labelsize=lsize)
    axes.tick_params(axis="x",labelsize=lsize)
    axes.set_ylim(-0.1,5.1)
    if log:
        axes.set_xscale("log")
        axes.set_xlim(win[0] * 0.75, win[-1]*1.5)
    axes.yaxis.set_major_locator(MaxNLocator(integer=True))
    axes2 = axes.twiny()
    axes2.set_xlabel(r"Time resolution $\Delta t$ [frames]",fontsize=fontsize)
    axes2.tick_params(labelsize=lsize)
    if log:
        axes2.set_xscale("log")
        axes2.set_xlim(win[0] * 0.75 / 0.1, win[-1]*1.5 / 0.1)
    axesr = axes.twinx()
    axesr.plot(win, env0, marker="o", c="#ff7f0e")
    axesr.set_ylim(-0.03,1.03)
    axesr.set_ylabel("ENV$_{0}$ fraction", weight="bold", c="#ff7f0e",fontsize=fontsize)
    axesr.tick_params(axis="y",colors="#ff7f0e",width=width,labelsize=lsize)
    # change all spines
    for axis in ['left','right']:
        axesr.spines[axis].set_linewidth(width)
    axesr.spines['left'].set_color("#1f77b4")
    axesr.spines['right'].set_color("#ff7f0e")
    #fig.show()
    fig.tight_layout()
    fig.savefig(f"{plot_position}.png")

def clean_pop(onion_folder, threshold):
    tau_window = []
    fraction_env0 = []
    with open(f"{onion_folder}/fraction_0.txt", 'r') as file:
        for line in file:
            if(line.startswith("#")):
                continue
            values = line.split()  
            if values:
                tau_window.append(float(values[0]))  
                fraction_env0.append(float(values[1]))
    
    states_pop = {i: [] for i in tau_window}
    with open(f"{onion_folder}/final_states.txt", 'r') as file:
        a = -1
        for line in file:
            if(line.startswith("##")):
                a += 1
                continue
            if(line.startswith("# ")):
                continue
            values = line.split()
            if values:
                try:
                    states_pop[tau_window[a]].append(float(values[3]))  
                except IndexError:
                    break
    n_states = []
    with open(f"{onion_folder}/number_of_states.txt", 'r') as file:
        for line in file:
            if(line.startswith("#")):
                continue
            values = line.split()
            if values:
                n_states.append(int(values[1]))
    print("tau_window:", tau_window)
    print("fraction in ENV0:", fraction_env0)
    print("number of states:", n_states)
    print("states pop:", states_pop)
    print("---")
    a = 0
    clean_states_pop = {i: [] for i in tau_window}
    for key, value in states_pop.items():
        for i in range(len(value)):
            if(value[i] < threshold and n_states[a] > 1):
                print(f"win: {key}: LOW POPULATION DETECTED")
                fraction_env0[a] += value[i]
                n_states[a] -= 1
                continue
            clean_states_pop[key].append(value[i])
        a += 1
        
    # Debug
    print("tau_window:", tau_window)
    print("fraction in ENV0:", fraction_env0)
    print("number of states:", n_states)
    print("states pop:", states_pop)
    print("clean states pop:", clean_states_pop)
    tau_window = np.array(tau_window)
    fraction_env0 = np.array(fraction_env0)
    n_states = np.array(n_states)
    return tau_window, fraction_env0, n_states

def des_of_des(n_states, fraction_env0):
    D = n_states * (1-fraction_env0)
    return D

directory = "arrays"
descriptors = []
for filename in os.listdir(directory):
    if(filename.endswith("SOAP_10.npy")):
        continue
    if os.path.isfile(os.path.join(directory, filename)):
        descriptors.append(filename)
for d in descriptors:
    print(d)
print("----")
for d in descriptors:
    plot_position = f"results/{d[:-4]}"
    tau_window, fraction_env0, n_states = clean_pop(f"onion/{d[:-4]}", 0.01)
    #[:-2]
    onion_plot(plot_position, tau_window, fraction_env0, n_states,True)
    D = des_of_des(n_states, fraction_env0)
    np.save(f"results/D_arrays/{d}", D)
    np.save(f"results/ST_arrays/{d}", n_states)
    np.save(f"results/ENV_arrays/{d}", fraction_env0)
np.save(f"results/D_arrays/windows_list", tau_window)
# %%
