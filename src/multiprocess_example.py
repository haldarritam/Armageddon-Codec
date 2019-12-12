import multiprocessing
import numpy as np
from random import randint

# The function that is going to run in parallel
# Notice that it can have as many parameters as we want
def worker_function(procnum, input_matrix, return_dict):
    random_number = randint(0,100)

    print("Hi, this is process " + str(procnum) + 
    " I am going to multiply the matrix of ones by " + 
    str(random_number) + "\n")

    result = random_number * input_matrix

    return_dict[procnum] = result


if __name__ == '__main__':
    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []

    input_matrix = np.ones((5,5), dtype=int)

    # This will create 5 copies of the worker_function running in parallel.
    # For the encoder, because we only need two process in parallel, we could
    # simply create p1 and p2 and do jobs.append(p1) jobs.append(p2) and p1.start()
    # and p2.start(). Later, we just need to do the join with the for structure.
    for i in range(5):
        p = multiprocessing.Process(target=worker_function, args=(i, input_matrix, return_dict)) # Passing parameters to the function
        jobs.append(p)
        p.start()

    for proc in jobs:
        proc.join()

    print(return_dict[0])
    print(return_dict[1])
    print(return_dict[2])
    print(return_dict[3])
    print(return_dict[4])


# For our encoder, the worker_function can simply be the block_encoding function. 
# The requirements are for just two threads, so we can run two instances of block_encoding at a time,
# and get the values ordered in the dictionary. Then, we extract results from the dictionary and merge
# them as needed to form the frame bitstream. 