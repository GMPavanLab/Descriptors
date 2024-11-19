#%%
import os
import shutil
from pathlib import Path

from onion_clustering import main

def perform_onion(path,
                  output_folder,
                  tau_w,
                  t_smooth,
                  t_delay,
                  t_conv,
                  time_units,
                  example_id,
                  num_tau_w,
                  min_tau_w,
                  min_t_smooth,
                  max_t_smooth,
                  step_t_smooth,
                  max_tau_w,
                  bins):
    ##############################################################################
    ### Set all the analysis parameters ###
    # Use git clone git@github.com:matteobecchi/onion_example_files.git
    # to download example datasets
    PATH_TO_INPUT_DATA = path
    TAU_WINDOW = tau_w  # time resolution of the analysis

    ### Optional parametrers ###
    T_SMOOTH = t_smooth  # window for moving average (default 1 - no average)
    T_DELAY = t_delay  # remove the first t_delay frames (default 0)
    T_CONV = t_conv  # convert frames in time units (default 1)
    TIME_UNITS = time_units  # the time units (default 'frames')
    EXAMPLE_ID = example_id  # particle plotted as example (default 0)
    NUM_TAU_W = num_tau_w  # number of values of tau_window tested (default 20)
    MIN_TAU_W = min_tau_w  # min number of tau_window tested (default 2)
    MIN_T_SMOOTH = min_t_smooth  # min value of t_smooth tested (default 1)
    MAX_T_SMOOTH = max_t_smooth  # max value of t_smooth tested (default 5)
    STEP_T_SMOOTH = step_t_smooth  # increment in value of t_smooth tested (default 1)
    MAX_TAU_W = max_tau_w  # max number of tau_window tested (default is auto)
    BINS = bins  # number of histogram bins (default is auto)
    ##############################################################################

    ### Create the output directory and move there ###
    original_dir = Path.cwd()
    output_path = Path(f"./{output_folder}")
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir()
    os.chdir(output_path)

    try:
        ### Create the 'data_directory.txt' file ###
        with open("data_directory.txt", "w+", encoding="utf-8") as file:
            print(PATH_TO_INPUT_DATA, file=file)

        ### Create the 'input_parameter.txt' file ###
        with open("input_parameters.txt", "w+", encoding="utf-8") as file:
            print(f"tau_window\t{TAU_WINDOW}", file=file)
            print(f"t_smooth\t{T_SMOOTH}", file=file)
            print(f"t_delay\t{T_DELAY}", file=file)
            print(f"t_conv\t{T_CONV}", file=file)
            print(f"t_units\t{TIME_UNITS}", file=file)
            print(f"example_ID\t{EXAMPLE_ID}", file=file)
            print(f"num_tau_w\t{NUM_TAU_W}", file=file)
            print(f"min_tau_w\t{MIN_TAU_W}", file=file)
            print(f"min_t_smooth\t{MIN_T_SMOOTH}", file=file)
            print(f"max_t_smooth\t{MAX_T_SMOOTH}", file=file)
            print(f"step_t_smooth\t{STEP_T_SMOOTH}", file=file)
            if MAX_TAU_W != "auto":
                print(f"max_tau_w\t{MAX_TAU_W}", file=file)
            if BINS != "auto":
                print(f"bins\t{BINS}", file=file)

        ### Perform the clustering analysis ###
        cl_ob = main.main(full_output = False)

        ### Plot the output figures in output_figures/ ###

        # Plots number of states and fraction_0 as a function of tau_window
        cl_ob.plot_tra_figure()
        cl_ob.plot_pop_fractions()

        # Plots the raw data
        cl_ob.plot_input_data("Fig0")

        # Plots the data with the clustering thresholds and Gaussians
        cl_ob.plot_cumulative_figure()

        # Plots the colored signal for the particle with `example_ID` ID
        cl_ob.plot_one_trajectory()

        # Plots the mean time sequence inside each state
        cl_ob.data.plot_medoids()

        # Plots the population of each state as a function of time
        cl_ob.plot_state_populations()

        # Plots the Sankey diagram between the input time_windows
        #cl_ob.sankey([0, 10, 20, 30, 40])

        # Writes the files for the visualization of the colored trj
        if os.path.exists("../../trajectory.xyz"):
            cl_ob.print_colored_trj_from_xyz("../../trajectory.xyz")
        else:
            cl_ob.print_labels()
    finally:
        os.chdir(original_dir)

directory = "arrays"
descriptors = []
for filename in os.listdir(directory):
    if(filename.endswith("SOAP_10.npy")):
        continue
    if os.path.isfile(os.path.join(directory, filename)):
        descriptors.append(filename)

d = "sp_10_LENS_10.npy"
tau = 8
perform_onion(path=f"../../arrays/{d}",
                output_folder=f"single/{d[:-4]}_{tau}",
                tau_w=tau,
                t_smooth=1,
                t_delay=1,
                t_conv=0.1,
                time_units="ns",
                example_id=0,
                num_tau_w=20,
                min_tau_w=2,
                min_t_smooth=1,
                max_t_smooth=0,
                step_t_smooth=1,
                max_tau_w="auto",
                bins="auto")

# %%
