import numpy as np
import matplotlib.pyplot as plt

colors = ['b', 'orange', 'g', 'r', 'Brown', 'm', 'y', 'k', 'Brown', 'ForestGreen']


def draw_const(data, mat_const, ax1, ax2, title1, title2):

    if np.shape(mat_const)[0] != np.shape(mat_const)[1]:
        print ("Constraints matrix must be squared")
        return ax1, ax2
    else:
        size = np.shape(mat_const)[0]

    aux_const = mat_const - np.identity(size)

    for i in range(size):

        for j in range(i + 1, size):

            if aux_const[i, j] == 1:

                ax1.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], linestyle = '-', color = "black")
                # ax1.annotate(str(i), (data[i, :]))
                # ax1.annotate(str(j), (data[j, :]))

            elif aux_const[i, j] == -1:

                ax2.plot([data[i, 0], data[j, 0]], [data[i, 1], data[j, 1]], linestyle = '--', color = "black")

    ax1.set_title(title1)
    ax2.set_title(title2)

    return ax1, ax2


def draw_data_2DNC(data, labels, numb_labels, title, ln = False):

    fig0, ax0 = plt.subplots()

    for label in range(numb_labels):
        ax0.plot(data[labels == label, 0], data[labels == label, 1], '.', color=colors[label])

    if ln:
        for i in range(data.shape[0]):
            ax0.annotate(str(labels[i]), data[i,:])

    ax0.set_title(title)
    ax0.set_xlabel("x")
    ax0.set_ylabel("y")
    return ax0

def draw_centroids(image, cent):

    image.plot(cent[1, :], cent[0, :], 'ro')

    return image
