import sys
from time import time
import multiprocessing as mp
import torch
from torch.utils.data import DataLoader

def find_num_workers(training_data, batch_size):
    best_time = sys.maxsize
    best_workers = 0
    for num_workers in range(0, mp.cpu_count(), 2):
        train_loader = DataLoader(training_data, shuffle=True, num_workers=num_workers, batch_size=batch_size,
                                  pin_memory=torch.cuda.is_available())
        start = time()
        for epoch in range(1, 3):
            for i, data in enumerate(train_loader, 0):
                pass
        end = time()
        tot_time = end - start
        print("Finish with:{} second, num_workers={}".format(tot_time, num_workers))
        if int(tot_time*100) < best_time:
            best_time = int(tot_time*100)
            best_workers = num_workers
    print("Optimal number of workers for this dataset and CPU/GPU configuration is %d workers" % best_workers)
    return best_workers

