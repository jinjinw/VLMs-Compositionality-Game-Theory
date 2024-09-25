from tqdm import tqdm
from random import random
from multiprocessing import Queue
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from concurrent.futures import as_completed


def runners(job, workers, runner, with_tqdm=False, world_size=1, **kwargs):
    max_workers = 10000
    num_workers = len(workers) * world_size
    with ThreadPoolExecutor(runner) as executor:
        if with_tqdm:
            with tqdm(total=num_workers) as pbar:
                for index in range(0, len(workers), max_workers):
                    futures = [executor.submit(job, worker, **kwargs) for worker in workers[index: index+max_workers]]
                    for future in as_completed(futures):
                        pbar.update(world_size)
                        yield future.result()
        else:
            for index in range(0, len(workers), max_workers):
                futures = [executor.submit(job, worker, **kwargs) for worker in workers[index: index+max_workers]]
                for future in as_completed(futures):
                    yield future.result()

def prunners(job, workers, runner, mode='futures', **kwargs):
    worker_num = len(workers)
    max_workers = 2000
    if mode == 'futures':
        with ProcessPoolExecutor(runner) as executor:
            with tqdm(total=worker_num) as pbar:
                for index in range(0, len(workers), max_workers):
                    futures = [executor.submit(job, worker, **kwargs) for worker in workers[index: index+max_workers]]
                    for future in as_completed(futures):
                        pbar.update(1)
                        yield future.result()
    else:
        if len(kwargs.keys()):
            raise RuntimeError("Only support iterables under map mode.")
        chunksize = worker_num // max_workers
        with ProcessPoolExecutor(runner) as executor:
            with tqdm(total=worker_num) as pbar:
                for result in executor.map(job, workers, chunksize=chunksize):
                    pbar.update(1)
                    yield result


def task(name, test_args=1):
    value = random()
    return f'Task={name}: {value:.2f}: {test_args}'


def main():
    results = prunners(
        task, [f'worker_{i}' for i in range(10000002)], 
        runner=10, mode='map'
    )
    for re in results:
        pass  # print(re)


if __name__ == '__main__':
    main()
