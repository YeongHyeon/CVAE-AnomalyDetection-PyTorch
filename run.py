import os, argparse
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch

import source.neuralnet as nn
import source.datamanager as dman
import source.solver as solver

def main():

    dataset = dman.Dataset(normalize=FLAGS.datnorm)

    if(not(torch.cuda.is_available())): FLAGS.ngpu = 0
    device = torch.device("cuda" if (torch.cuda.is_available() and FLAGS.ngpu > 0) else "cpu")

    neuralnet = nn.NeuralNet(height=dataset.height, width=dataset.width, channel=dataset.channel, \
        device=device, ngpu=FLAGS.ngpu, \
        ksize=FLAGS.ksize, z_dim=FLAGS.z_dim, learning_rate=FLAGS.lr)

    solver.training(neuralnet=neuralnet, dataset=dataset, epochs=FLAGS.epoch, batch_size=FLAGS.batch)
    solver.test(neuralnet=neuralnet, dataset=dataset)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngpu', type=int, default=1, help='-')
    parser.add_argument('--datnorm', type=bool, default=True, help='Data normalization')
    parser.add_argument('--ksize', type=int, default=3, help='kernel size for constructing Neural Network')
    parser.add_argument('--z_dim', type=int, default=128, help='Dimension of latent vector')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for training')
    parser.add_argument('--epoch', type=int, default=100, help='Training epoch')
    parser.add_argument('--batch', type=int, default=32, help='Mini batch size')

    FLAGS, unparsed = parser.parse_known_args()

    main()
