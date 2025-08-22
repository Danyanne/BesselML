from BesselML.main import HyperParameterSearch
from multiprocessing import Pool, cpu_count
import numpy as np
from scipy import special

def worker_process(dataset_name, X, y):
    """
    A top-level function that instantiates and runs the search for a dataset.
    This version accepts a dataset name and the data arrays (X, y).
    """
    searcher = HyperParameterSearch() 
    searcher.run_for_data(dataset_name, X, y)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # Generate the custom dataset
    order = 0
    # Use a single, combined feature array (X)
    X = np.sort(np.concatenate((np.random.uniform(1, 50, 300), np.linspace(1e-2, 3, 50)))).reshape(-1, 1)
    # The target array (y)
    y = special.jv(order, X).flatten()
    
    # Create a list of tuples for starmap
    # Each tuple contains the arguments for worker_process: (name, X, y)
    datasets_to_run = [('Bessel_J_0', X, y)]
    
    # Set the number of parallel processes
    N_PROCESSES = cpu_count() // 2

    print(f"Starting hyperparameter search for {len(datasets_to_run)} datasets using {N_PROCESSES} processes.")
    
    # Create a pool of worker processes
    with Pool(N_PROCESSES) as pool:
        # Use starmap, which unpacks each tuple in the iterable as arguments
        pool.starmap(worker_process, datasets_to_run)

    print("All experiments have been completed.")
