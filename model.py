import matplotlib.pyplot as plt
from math import floor, sqrt

import torch
from torch import nn
from torchvision.transforms.functional import to_pil_image
from torch.utils.data import DataLoader

from tqdm import tqdm


#Hyperparameters

learning_rate = 10e-6 #learning_rate
n_epochs = 300 #epochs for training
optimizer : torch.optim.Optimizer = torch.optim.Adam
n_filters = 6 #base number for the number of filters used by the generator

class DCGAN(nn.Module):
    '''
    Wrapper class for the DCGAN, offering image generation methods through its generator.
    Main parameters as global variables in model.py
    '''
    def __init__(self,
                 ) -> None:
        super().__init__()

        self.current_epoch : int = 1 # index of current epoch of training

        self.GENERATOR = GAN_Generator(dim_embedding = 6,
                                       dim_random = 3,
                                       n_filters = n_filters)
        
        self.DISCRIMINATOR = GAN_Discriminator()

        self.optimizer_GEN : torch.optim.Optimizer = optimizer(self.GENERATOR.parameters(),
                                                               lr = learning_rate)
        self.optimizer_DISC : torch.optim.Optimizer = optimizer(self.DISCRIMINATOR.parameters(),
                                                               lr = learning_rate)
    
    @classmethod
    def load_model(cls, filepath : str = 'current_saved_model.pth'):
        '''
        Loads model from passed string, loading state of discriminator and generator, with current epoch parameter
        '''
        checkpoint = torch.load(f = filepath)
        model = DCGAN()
        model.GENERATOR.load_state_dict(checkpoint['generator'])
        model.DISCRIMINATOR.load_state_dict(checkpoint['discriminator'])
        model.current_epoch = checkpoint['current_epoch']
        return model
    

    def gen_and_save(self, 
                     labels : torch.Tensor = None): #labels used for generation
        '''
        Generates images from passed labels (otherwise they are randomly generated) and saves them in gen_images
        '''
        if labels is None:
            labels : torch.Tensor = torch.randint(low = 0, high = 10, size = (16,)) # if no labels are passed, random ones are generated
        batched_images : torch.Tensor = self.GENERATOR.generate_random(labels = labels) #generate handwritten digits

        #Displays generated images in a square grid, discarding until reaching a perfect square
        n_images = batched_images.shape[0]
        side = floor(sqrt(n_images))
        fig, axs = plt.subplots(side,side)
        fig.tight_layout()
        
        for img_idx in range(side**2):
            img = batched_images[img_idx,:,:,:].squeeze(dim=0)
            axs[img_idx//side][img_idx%side].imshow(to_pil_image(img), cmap = 'gray', vmin= 0, vmax = 255)
            axs[img_idx//side][img_idx%side].set_xlabel(labels[img_idx].item())

        plt.savefig(f'gen_images/gen_images_{self.current_epoch}')
        plt.close()


    def gen_and_show(self):
        '''
        Generates images from passed labels (otherwise they are randomly generated) and displays them
        '''
        labels : torch.Tensor = torch.randint(low = 0, high = 10, size = (16,))
        batched_images : torch.Tensor = self.GENERATOR.generate_random(labels = labels)

        n_images = batched_images.shape[0]
        side = floor(sqrt(n_images))
        fig, axs = plt.subplots(side,side)
        fig.tight_layout()
        
        for img_idx in range(side**2):
            img = batched_images[img_idx,:,:,:].squeeze(dim=0)
            axs[img_idx//side][img_idx%side].imshow(to_pil_image(img), cmap = 'gray', vmin= 0, vmax = 255)
            axs[img_idx//side][img_idx%side].set_xlabel(labels[img_idx].item())

        plt.show()
        plt.close()

        

    def train(self,
              TrainLoader : DataLoader):
        '''
        Trains the DCGAN from a DataLoader.

        Convention : 
        disc ouputs 1 -> image is classified as real
        disc ouputs 0 -> image is classified as fake
        '''
        
        batch_size : int = TrainLoader.batch_size

        self.GENERATOR.train(mode = True)
        self.DISCRIMINATOR.train(mode = True)

        for epoch in range(1, n_epochs+1):

            print(10*'-'+f'[ Epoch {epoch} ]'+10*'-')

            loop = tqdm(TrainLoader)
            
            for batch_idx, (true_images, true_labels) in enumerate(loop):
                
                #Train Generator
                fake_labels : torch.Tensor = torch.randint(low = 0, high = 10, size = (batch_size,))
                fake_images : torch.Tensor = self.GENERATOR.generate_random(labels = fake_labels)

                prob_fakes : torch.Tensor = self.DISCRIMINATOR.discriminate(batched_images = fake_images,
                                                             labels = fake_labels)
                
                mean_prob_fakes : torch.Tensor = prob_fakes.mean()
                
                loss_gen : torch.Tensor = - torch.log(mean_prob_fakes)

                self.optimizer_GEN.zero_grad()
                loss_gen.backward()
                self.optimizer_GEN.step()

                #Train Discriminator
                fake_labels : torch.Tensor = torch.randint(low = 0, high = 10, size = (batch_size,))
                fake_images : torch.Tensor = self.GENERATOR.generate_random(labels = fake_labels)

                prob_fakes : torch.Tensor = self.DISCRIMINATOR.discriminate(batched_images = fake_images,
                                                             labels = fake_labels)
                prob_reals : torch.Tensor = self.DISCRIMINATOR.discriminate(batched_images = true_images,
                                                                            labels = true_labels)
                
                mean_prob_fakes : torch.Tensor = prob_fakes.mean()
                mean_prob_reals : torch.Tensor = prob_reals.mean()

                coef = ((1 / mean_prob_fakes)-1).item() # discutable use of learning statistics to ajust the loss, turns out to work in practice

                loss_disc : torch.Tensor = torch.log(mean_prob_fakes) - coef * torch.log(mean_prob_reals) 

                self.optimizer_DISC.zero_grad()
                loss_disc.backward()
                self.optimizer_DISC.step()

                loop.set_description(f'P_FAKES: {mean_prob_fakes.item()}, P_REALS: {mean_prob_reals.item()}')

            if mean_prob_fakes.item()!= torch.nan:
                torch.save(obj = {
                    'generator' : self.GENERATOR.state_dict(),
                    'discriminator' : self.DISCRIMINATOR.state_dict(),
                    'current_epoch' : self.current_epoch
                },
                f = 'current_saved_model.pth')


            self.gen_and_save(epoch = epoch)
            self.current_epoch += 1
                

                

class GAN_Generator(nn.Module):

    def __init__(self,
                 dim_embedding : int,
                 dim_random : int,
                 n_filters : int):
        super().__init__()

        self.dim_embedding : int = dim_embedding
        self.dim_random : int = dim_random

        self.embedding : torch.nn.Embedding = nn.Embedding(num_embeddings = 10,
                                      embedding_dim = self.dim_embedding)

        self.ConvTransp2D : torch.nn.Module = nn.Sequential(

            nn.ConvTranspose2d(
                in_channels = self.dim_embedding+self.dim_random,
                out_channels = 8*n_filters,
                kernel_size = 4,
            ),
            nn.BatchNorm2d(8*n_filters),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(
                in_channels =  8*n_filters,
                out_channels =  4*n_filters,
                kernel_size = 4,
                stride = 2,
            ),
            nn.BatchNorm2d(4*n_filters),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(
                in_channels =  4*n_filters,
                out_channels =  2*n_filters,
                kernel_size = 4,
                stride = 2
            ),
            nn.BatchNorm2d(2*n_filters),
            nn.ReLU(inplace = True),

            nn.ConvTranspose2d(
                in_channels =  2*n_filters,
                out_channels =  1,
                kernel_size = 7,
                stride = 1,
            ),
            nn.Tanh(),
            nn.ReLU())


    def forward(self, x) -> torch.Tensor:
        return self.ConvTransp2D(x)
    

    def generate_random(self,
                        labels : torch.Tensor = None) -> torch.Tensor:
        '''
        Generates random images according to passed labels. If no labels are passed, they are generated randomly.
        Returns batched images.
        '''
        
        if labels is None:
            labels = torch.randint(low = 0,
                                   high = 10,
                                   size = (32,))
            
        embedded_labels : torch.Tensor = self.embedding(labels) #computes embedding of the labels
        seeds : torch.Tensor = torch.randn((embedded_labels.shape[0],self.dim_random)) #generates random values

        input : torch.Tensor = torch.cat(tensors = (embedded_labels, seeds),dim = 1) #concatenates embedded labels and random seed along channel dimension
        input = input.reshape((embedded_labels.shape[0],
                               self.dim_embedding + self.dim_random,
                               1,
                               1,
                               )) #reshpae to (batch,dim_embed+dim_random,1,1)
        
        return self.ConvTransp2D(input) #generates images and returns
    

class GAN_Discriminator(nn.Module):

    def __init__(self):
        super().__init__()
        self.embedding : nn.Embedding = nn.Embedding(num_embeddings = 10,
                                           embedding_dim = 28*28)
        self.CNN : nn.Module = nn.Sequential(
            nn.Conv2d(in_channels = 2,
                    out_channels = 32,
                    kernel_size = 5,
                    stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(in_channels = 32,
                    out_channels = 64,
                    kernel_size = 5,
                    stride = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Flatten(),

            nn.Linear(in_features = 1024,
                    out_features = 128),
            nn.ReLU(),
            nn.Linear(in_features = 128,
                    out_features = 1),
            nn.Sigmoid()
        )
    

    def forward(self, x):
        return self.CNN(x)
    

    def discriminate(self,
                     batched_images : torch.Tensor,
                     labels : torch.Tensor) -> torch.Tensor:
        '''
        Takes batched images with corresponding labels and feeds them into the 
        discriminator and returns probabilities
        '''
        
        embedded_labels : torch.Tensor = self.embedding(labels) #computes image embedding of labels
        embedded_labels = embedded_labels.reshape(batched_images.shape) # reshape to same shape as image
        input = torch.cat((batched_images, embedded_labels), dim = 1) #concatenates along channel dimension
        return self.CNN(input) #computes probs and returns
    