# DCGAN on MNIST dataset

In this project we build a conditional GAN on the MNIST dataset, able to generate new hand-written inspired from the ones given as input data.

In the main.py file we give ready-to-use functions which trigger training, loading or generating functions. See file for further details.

Loading : Two pre-trained models are featured GAN_3 and GAN_6. To load GAN_n, one should first set hidden_layers to n in the model.py file hyperparameters section. Otherwise they will not load.