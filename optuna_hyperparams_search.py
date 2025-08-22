from main import HyperParameterSearch

def worker_process(dataset_name):
    """A top-level function that instantiates and runs the search for a dataset."""
    # You can configure the searcher here if you want different settings per worker
    searcher = HyperParameterSearch() 
    searcher.run_for_dataset(dataset_name)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

if __name__ == '__main__':
    # List of datasets from PMLB to run the experiments on
    dataset_names = []
    
    # Set the number of parallel processes
    N_PROCESSES = cpu_count() - 3

    print(f"Starting hyperparameter search for {len(dataset_names)} datasets using {N_PROCESSES} processes.")
    
    # Create a pool of worker processes
    with Pool(N_PROCESSES) as pool:
        # The map function will call `worker_process` for each dataset name in the list.
        # Each call runs in a separate process in the pool.
        pool.map(worker_process, dataset_names)

    print("All experiments have been completed.")
