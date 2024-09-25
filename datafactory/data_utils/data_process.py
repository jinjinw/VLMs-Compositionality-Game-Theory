class DataProcess(object):
    def __init__(self, root_folder) -> None:
        self.root_folder = root_folder

    def preprocess(self, data_path) -> tuple:
        metainfo = {'label': 0, 'filename': f'random/test_case_{data_path}.png'}
        data = 'torch.ones(1, 3, 224, 224)'
        return metainfo, data

    def load_dataset(self, ) -> list:
        return [i for i in range(100)]
