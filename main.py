# coding=utf-8

import numpy as np
from mnist import load_mnist
from network import SimpleConvNet
from trainer import Trainer
import matplotlib.pyplot as plt


(x_train, t_train), (x_test, t_test) = load_mnist()

epochs = 5

network = SimpleConvNet(input_dim=(1, 28, 28),
                        conv_params={'filter_num': 10, 'filter_size': 5, 'stride': 1, 'pad': 0},
                        hiddensize=100,
                        output_size=10,
                        weight_init_std=0.01)

trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs, batch_size=32, optimizer='SGD', optimizer_params={'lr': 0.001})

trainer.train()

markers = {'train': 'o', 'test': 's'}
x = np.arange(epochs)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()