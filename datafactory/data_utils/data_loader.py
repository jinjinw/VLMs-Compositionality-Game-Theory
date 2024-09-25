from collections import defaultdict

from datafactory.utils import dataprocess_init
from datafactory.runner import prunners, runners


def get_processed_data(inputs, dataprocess_file):
    prf = dataprocess_init(dataprocess_file).preprocess
    return prf(inputs)


class BaseDatasetLoader(object):
    """Base Dataset Loader."""
    def __init__(
        self, dataprocess, dataprocess_file=None, 
        global_rank=0, world_size=1, 
        batch_size=64, runner=6, runner_mode='multi_threads'
    ):
        super().__init__()
        
        self.dataset = None
        self.batch_size = batch_size

        # data process file init.
        self.data_process = dataprocess
        # load dataset with root_folder.
        self.dataset = self.load_dataset(global_rank, world_size)
        self.dataset_len = self._get_datasetlen()

        self.atoms_iter = self.atom_generator(
            runner=runner, dataprocess_file=dataprocess_file, mode=runner_mode,
            world_size=world_size, global_rank=global_rank, 
        )
        self.dataiter = self.generate_batch_data()

    def _get_datasetlen(self) -> int:
        return len(self.dataset)

    def load_dataset(self, global_rank, world_size):
        subprocess_dataset = list()
        for index, datameta in enumerate(self.data_process.load_dataset()):
            if index % world_size == global_rank:
                subprocess_dataset.append(datameta)
        return subprocess_dataset
    
    def atom_generator(self, runner, dataprocess_file, mode='multi_threads', world_size=1, global_rank=0):
        if mode == 'multi_threads':
            with_tqdm = True if global_rank == 0 else False
            return runners(
                self.data_process.preprocess, list(self.dataset), runner=runner, 
                with_tqdm=with_tqdm, world_size=world_size)
                
        return prunners(
            get_processed_data, list(self.dataset), runner=runner, 
            dataprocess_file=dataprocess_file
        )

    def generate_batch_data(self,):
        while True:
            batch_datas = defaultdict(list)
            sample_num = 0
            while sample_num < self.batch_size:
                try:
                    atom_tuple = next(self.atoms_iter)
                    data, errmsg = atom_tuple
                    if data is None:
                        print(errmsg)
                    else:
                        for k, v in data.items():
                            batch_datas[k].append(v)
                        sample_num += 1
                except StopIteration:
                    break
            if len(batch_datas) == 0:
                break 
            yield batch_datas
