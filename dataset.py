from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from torch.utils.data import DataLoader


class DATAWRAPPER():

    def __init__(self,
                 download = True,
                 ):
        
        self.DATA_TRAIN = MNIST('data',
                                train = True,
                                download = download,
                                transform = ToTensor())
        
        self.DATA_TEST = MNIST('data',
                               train = False,
                               download = download,
                               transform = ToTensor())


    def get_LOADER_TRAIN(self, batch_size = 32):
        return DataLoader(self.DATA_TRAIN, batch_size = batch_size, shuffle = True)
    

    def get_LOADER_TEST(self, batch_size = 1):
        return DataLoader(self.DATA_TEST, batch_size = batch_size, shuffle = True)
    