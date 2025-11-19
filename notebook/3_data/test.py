#!/usr/bin/env python3
import time
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each rank prints its own message
    time.sleep(10.0)
    print(f"Hello from rank {rank} of {size}")

    # Optionally, have rank 0 gather something
    msg = f"rank {rank} reporting in"
    gathered = comm.gather(msg, root=0)

    if rank == 0:
        print("Rank 0 gathered:")
        for g in gathered:
            print("  ", g)

if __name__ == "__main__":
    main()
