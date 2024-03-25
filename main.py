from model import DCGAN
from dataset import DATAWRAPPER


def mainTrain():
    '''
    Creates new GAN and trains it. Model is periodically saved, and examples of generated images are stored in gen_images repo
    '''
    model = DCGAN()
    datawrapper = DATAWRAPPER()
    TrainLoader = datawrapper.get_LOADER_TRAIN(batch_size = 32)
    model.train(TrainLoader = TrainLoader)

def loadTrain(filepath : str):
    '''
    Loads a pretrained model from location from current repo, passed as string and continues training
    '''
    model = DCGAN.load_model(filepath = filepath)
    datawrapper = DATAWRAPPER()
    TrainLoader = datawrapper.get_LOADER_TRAIN(batch_size = 32)
    model.train(TrainLoader = TrainLoader)

def loadGen(filepath : str):
    '''
    Loads pretrained model from filepath and generates new digits
    '''
    model = DCGAN.load_model(filepath = filepath)
    model.gen_and_show()

if __name__ == '__main__':
    loadTrain()