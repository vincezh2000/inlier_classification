import numpy as np
import matplotlib.pyplot as plt


def plot_loss(loss,no_epochs):
    '''
    plot loss function
    '''

    epochs=np.arange(no_epochs)
    plt.plot(loss,no_epochs)
    plt.show()

