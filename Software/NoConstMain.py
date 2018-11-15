import numpy as np
from sklearn import datasets
from AGNoConst.BRKGA import BRKGA
from functions import draw_data_2DNC, generate_data_2D
import matplotlib.pyplot as plt


def main():
    np.random.seed(43)
    random_state = 43
    const_percent = 0.6
    alg = "BRKGA"

    iris = datasets.load_iris()
    iris_set = iris.data[:, :2]
    iris_labels = iris.target
    rand_set, rand_labels = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 50)

    iris_brkga = BRKGA(iris_set, 1000, 0.2, 0.2, 0.5, 3)

    iris_cekm_assignment = iris_brkga.run(500)

    iris_plot = draw_data_2DNC(iris_set, np.asarray(iris_cekm_assignment, np.float), 3, alg + "Iris Dataset Results")

    plt.show()


if __name__ == "__main__": main()