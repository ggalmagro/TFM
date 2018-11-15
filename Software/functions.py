import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn import datasets
import itertools
import random


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


def generate_data_2D(centers, sigmas, numb_data):

    xpts = np.zeros(1)
    ypts = np.zeros(1)
    labels = np.zeros(1)
    for i, ((xmu, ymu), (xsigma, ysigma)) in enumerate(zip(centers, sigmas)):
        xpts = np.hstack((xpts, np.random.standard_normal(numb_data) * xsigma + xmu))
        ypts = np.hstack((ypts, np.random.standard_normal(numb_data) * ysigma + ymu))
        labels = np.hstack((labels, np.ones(numb_data) * i))

    X = np.zeros((len(xpts) - 1, 2))
    X[:, 0] = xpts[1:]
    X[:, 1] = ypts[1:]

    y = labels[1:]

    return X, y


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

def gen_rand_const(labels, nb_const):

    pairs = np.array(list(itertools.combinations(range(0, len(labels)), 2)))
    ind = random.sample(range(0, len(pairs)), nb_const)
    const = pairs[ind]

    const_matrix = np.identity(len(labels))

    for i in const:

        if labels[i[0]] == labels[i[1]]:

            const_matrix[i[0], i[1]] = 1
            const_matrix[i[1], i[0]] = 1

        else:

            const_matrix[i[0], i[1]] = -1
            const_matrix[i[1], i[0]] = -1

    return const_matrix

def gen_rand_const2(X, y, mat_const, nb_const, noise = 0, prop = False):

    n, nb_att = np.shape(X)

    # check the number of constraint to find
    nb_const_max = math.factorial(n - 1)
    nb_const_actual = np.shape(np.nonzero(mat_const - np.identity(n)))[1]

    if nb_const_max < nb_const_actual - nb_const:

        nb_const = nb_const_max - nb_const_actual

        if np.sign(nb_const) == -1:
            print("Error: number of constraints is too high")
            return -1

    i = 0

    while (i < nb_const):

        alredy_used = False
        ind_mat1, ind_mat2 = np.where(mat_const - np.identity(n) != 0)

        ind1 = np.random.randint(0, n)
        ind2 = np.random.randint(0, n)

        while (ind1 == ind2):
            ind2 = np.random.randint(0, n)

        where1 = np.where(ind_mat1 == ind1)[0]
        where2 = np.where(ind_mat2 == ind1)[0]
        test_ind1 = np.concatenate((where1, where2), axis = 0)

        if len(test_ind1) != 0:  # if the new indice 1 is alredy used

            test_ind2 = np.concatenate((np.where(ind_mat2[test_ind1] == ind2)[0], np.where(ind_mat1[test_ind1] == ind2)[0]))

            if len(test_ind2) != 0:  # if the new pair is alredy used

                alredy_used = True

        if (not alredy_used):

            mat_const[ind1, ind2] = (y[ind1] == y[ind2]) * 2 - 1

            if (noise > np.random.rand()):
                mat_const[ind1, ind2] = int((mat_const[ind1, ind2] == -1)) * 2 - 1
            i += 1

    result_matrix = mat_const

    if (prop == 1):

        aux1 = np.sign(mat_const + mat_const.conj().T - np.identity(n))
        aux2 = mat_const * np.sign(mat_const)
        aux2 = np.maximum(aux2, aux2.conj().T)
        mat_const = aux1 * aux2

        # Must link propagation
        ml = (mat_const > 0)

        for i in range(1, nb_const):
            ml = np.matrix(ml.tolist() or np.linalg.matrix_power(ml, i))

        # Cannot link propagation
        cl = (mat_const < 0) * 1
        cl = np.matrix((cl.tolist() or np.dot(cl, ml).tolist or np.dot(ml, cl) * 1))
        cl = np.matrix(cl.tolist() or np.dot(cl, ml).tolist())

        result_matrix = ml - cl

    return result_matrix


def get_const_list(m):
    ml = []
    cl = []

    for i in range(np.shape(m)[0]):
        for j in range(i + 1, np.shape(m)[0]):
            if m[i, j] == 1:
                ml.append((i, j))
            if m[i, j] == -1:
                cl.append((i, j))

    return ml, cl


def twospirals(n_points, noise=.5):
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))), np.hstack((np.zeros(n_points),np.ones(n_points))))


def get_usat_percent(ml, cl, clustering):

    unsat = 0

    for i in range(len(ml)):

        if clustering[ml[i][0]] != clustering[ml[i][1]]:
            unsat += 1

    for i in range(len(cl)):

        if clustering[cl[i][0]] == clustering[cl[i][1]]:
            unsat += 1

    return (unsat / (len(ml) + len(cl))) * 100


def load_all_datasets():

    random_state = 43
    datasets_array = []
    labels_array = []
    names_array = []

    # CARGA DE DATOS

    iris = datasets.load_iris()
    iris_set = iris.data[:, :2]
    iris_labels = iris.target
    datasets_array.append(iris_set)
    labels_array.append(iris_labels)
    names_array.append("iris")

    wine = datasets.load_wine()
    wine_set = wine.data
    wine_labels = wine.target
    datasets_array.append(wine_set)
    labels_array.append(wine_labels)
    names_array.append("wine")

    soybean = np.loadtxt("Datasets/soybean-small.data.txt", delimiter=',', dtype=str)
    soybean_labels = np.asarray(soybean[:, 35])
    soybean_set = np.asarray(soybean[:, :35].astype(float))
    datasets_array.append(soybean_set)
    labels_array.append(soybean_labels)
    names_array.append("soybean")

    # breastA_set = np.transpose(np.loadtxt("Datasets/Breast_A.gct.txt", skiprows=3, usecols=np.array(range(2, 100))))
    # breastA_labels = np.asarray(np.loadtxt("Datasets/Breast_A.cls", skiprows=2))
    # datasets_array.append(breastA_set)
    # labels_array.append(breastA_labels)
    # names_array.append("breastA")

    pima = np.loadtxt("Datasets/pima.dat", skiprows=13, delimiter=',', dtype=str)
    pima_labels = pima[:, 8]
    pima_set = np.asarray(pima[:, :8].astype(float))
    datasets_array.append(pima_set)
    labels_array.append(pima_labels)
    names_array.append("pima")

    balance = np.loadtxt("Datasets/balance.dat", skiprows=9, delimiter=',', dtype=str)
    balance_labels = balance[:, 4]
    balance_set = np.asarray(balance[:, :4].astype(float))
    datasets_array.append(balance_set)
    labels_array.append(balance_labels)
    names_array.append("balance")

    bupa = np.loadtxt("Datasets/bupa.dat", skiprows=11, delimiter=',', dtype=str)
    bupa_labels = bupa[:, 6]
    bupa_set = np.asarray(bupa[:, :6].astype(float))
    datasets_array.append(bupa_set)
    labels_array.append(bupa_labels)
    names_array.append("bupa")

    contraceptive = np.loadtxt("Datasets/contraceptive.dat", skiprows=14, delimiter=',', dtype=str)
    contraceptive_labels = contraceptive[:, 9]
    contraceptive_set = np.asarray(contraceptive[:, :9].astype(float))
    datasets_array.append(contraceptive_set)
    labels_array.append(contraceptive_labels)
    names_array.append("contraceptive")

    ecoli = np.loadtxt("Datasets/ecoli.dat", skiprows=12, delimiter=',', dtype=str)
    ecoli_labels = ecoli[:, 7]
    ecoli_set = np.asarray(ecoli[:, :7].astype(float))
    datasets_array.append(ecoli_set)
    labels_array.append(ecoli_labels)
    names_array.append("ecoli")

    haberman = np.loadtxt("Datasets/haberman.dat", skiprows=8, delimiter=',', dtype=str)
    haberman_labels = haberman[:, 3]
    haberman_set = np.asarray(haberman[:, :3].astype(float))
    datasets_array.append(haberman_set)
    labels_array.append(haberman_labels)
    names_array.append("haberman")

    led7digit = np.loadtxt("Datasets/led7digit.dat", skiprows=12, delimiter=',', dtype=str)
    led7digit_labels = led7digit[:, 7]
    led7digit_set = np.asarray(led7digit[:, :7].astype(float))
    datasets_array.append(led7digit_set)
    labels_array.append(led7digit_labels)
    names_array.append("led7digit")

    monk2 = np.loadtxt("Datasets/monk-2.dat", skiprows=11, delimiter=',', dtype=str)
    monk2_labels = monk2[:, 6]
    monk2_set = np.asarray(monk2[:, :6].astype(float))
    datasets_array.append(monk2_set)
    labels_array.append(monk2_labels)
    names_array.append("monk2")

    newthyroid = np.loadtxt("Datasets/newthyroid.dat", skiprows=10, delimiter=',', dtype=str)
    newthyroid_labels = newthyroid[:, 5]
    newthyroid_set = np.asarray(newthyroid[:, :5].astype(float))
    datasets_array.append(newthyroid_set)
    labels_array.append(newthyroid_labels)
    names_array.append("newthyroid")

    # thyroid = np.loadtxt("Datasets/thyroid.dat", skiprows=26, delimiter=',', dtype=str)
    # thyroid_labels = thyroid[:, 21]
    # thyroid_set = np.asarray(thyroid[:, :21].astype(float))
    # datasets_array.append(thyroid_set)
    # labels_array.append(thyroid_labels)
    # names_array.append("thyroid")

    vehicle = np.loadtxt("Datasets/vehicle.dat", skiprows=23, delimiter=',', dtype=str)
    vehicle_labels = vehicle[:, 18]
    vehicle_set = np.asarray(vehicle[:, :18].astype(float))
    datasets_array.append(vehicle_set)
    labels_array.append(vehicle_labels)
    names_array.append("vehicle")

    zoo = np.loadtxt("Datasets/zoo.dat", skiprows=21, delimiter=',', dtype=str)
    zoo_labels = zoo[:, 16]
    zoo_set = np.asarray(zoo[:, :16].astype(float))
    datasets_array.append(zoo_set)
    labels_array.append(zoo_labels)
    names_array.append("zoo")

    sonar = np.loadtxt("Datasets/sonar.dat", skiprows=65, delimiter=',', dtype=str)
    sonar_labels = sonar[:, 60]
    sonar_set = np.asarray(sonar[:, :60].astype(float))
    datasets_array.append(sonar_set)
    labels_array.append(sonar_labels)
    names_array.append("sonar")

    heart = np.loadtxt("Datasets/heart.dat", skiprows=18, delimiter=',', dtype=str)
    heart_labels = heart[:, 13]
    heart_set = np.asarray(heart[:, :13].astype(float))
    datasets_array.append(heart_set)
    labels_array.append(heart_labels)
    names_array.append("heart")

    ionosphere = np.loadtxt("Datasets/ionosphere.dat", skiprows=38, delimiter=',', dtype=str)
    ionosphere_labels = ionosphere[:, 33]
    ionosphere_set = np.asarray(ionosphere[:, :33].astype(float))
    datasets_array.append(ionosphere_set)
    labels_array.append(ionosphere_labels)
    names_array.append("ionosphere")

    wdbc = np.loadtxt("Datasets/wdbc.dat", skiprows=35, delimiter=',', dtype=str)
    wdbc_labels = wdbc[:, 30]
    wdbc_set = np.asarray(wdbc[:, :30].astype(float))
    datasets_array.append(wdbc_set)
    labels_array.append(wdbc_labels)
    names_array.append("wdbc")

    vowel = np.loadtxt("Datasets/vowel.dat", skiprows=18, delimiter=',', dtype=str)
    vowel_labels = vowel[:, 13]
    vowel_set = np.asarray(vowel[:, :13].astype(float))
    datasets_array.append(vowel_set)
    labels_array.append(vowel_labels)
    names_array.append("vowel")

    movement_libras = np.loadtxt("Datasets/movement_libras.dat", skiprows=95, delimiter=',', dtype=str)
    movement_libras_labels = movement_libras[:, 90]
    movement_libras_set = np.asarray(movement_libras[:, :90].astype(float))
    datasets_array.append(movement_libras_set)
    labels_array.append(movement_libras_labels)
    names_array.append("movement_libras")

    appendicitis = np.loadtxt("Datasets/appendicitis.dat", skiprows=12, delimiter=',', dtype=str)
    appendicitis_labels = appendicitis[:, 7]
    appendicitis_set = np.asarray(appendicitis[:, :7].astype(float))
    datasets_array.append(appendicitis_set)
    labels_array.append(appendicitis_labels)
    names_array.append("appendicitis")

    saheart = np.loadtxt("Datasets/saheart.dat", skiprows=14, delimiter=',', dtype=str)
    saheart[saheart == "Present"] = "1"
    saheart[saheart == "Absent"] = "0"
    saheart_labels = saheart[:, 9]
    saheart_set = np.asarray(saheart[:, :9].astype(float))
    datasets_array.append(saheart_set)
    labels_array.append(saheart_labels)
    names_array.append("saheart")

    spectfheart = np.loadtxt("Datasets/spectfheart.dat", skiprows=49, delimiter=',', dtype=str)
    spectfheart_labels = spectfheart[:, 44]
    spectfheart_set = np.asarray(spectfheart[:, :44].astype(float))
    datasets_array.append(spectfheart_set)
    labels_array.append(spectfheart_labels)
    names_array.append("spectfheart")

    hayesroth = np.loadtxt("Datasets/hayes-roth.dat", skiprows=9, delimiter=',', dtype=str)
    hayesroth_labels = hayesroth[:, 4]
    hayesroth_set = np.asarray(hayesroth[:, :4].astype(float))
    datasets_array.append(hayesroth_set)
    labels_array.append(hayesroth_labels)
    names_array.append("hayesroth")

    tae = np.loadtxt("Datasets/tae.dat", skiprows=10, delimiter=',', dtype=str)
    tae_labels = tae[:, 5]
    tae_set = np.asarray(tae[:, :5].astype(float))
    datasets_array.append(tae_set)
    labels_array.append(tae_labels)
    names_array.append("tae")

    ####################################################################################

    satimage = np.loadtxt("Datasets/satimage.dat", skiprows=41, delimiter=',', dtype=str)
    satimage_labels = satimage[:, 36]
    satimage_set = np.asarray(satimage[:, :36].astype(float))
    datasets_array.append(satimage_set)
    labels_array.append(satimage_labels)
    names_array.append("satimage")

    texture = np.loadtxt("Datasets/texture.dat", skiprows=45, delimiter=',', dtype=str)
    texture_labels = texture[:, 40]
    texture_set = np.asarray(texture[:, :40].astype(float))
    datasets_array.append(texture_set)
    labels_array.append(texture_labels)
    names_array.append("texture")

    pageblocks = np.loadtxt("Datasets/page-blocks.dat", skiprows=15, delimiter=',', dtype=str)
    pageblocks_labels = pageblocks[:, 10]
    pageblocks_set = np.asarray(pageblocks[:, :10].astype(float))
    datasets_array.append(pageblocks_set)
    labels_array.append(pageblocks_labels)
    names_array.append("pageblocks")

    phoneme = np.loadtxt("Datasets/phoneme.dat", skiprows=10, delimiter=',', dtype=str)
    phoneme_labels = phoneme[:, 5]
    phoneme_set = np.asarray(phoneme[:, :5].astype(float))
    datasets_array.append(phoneme_set)
    labels_array.append(phoneme_labels)
    names_array.append("phoneme")

    segment = np.loadtxt("Datasets/segment.dat", skiprows=24, delimiter=',', dtype=str)
    segment_labels = segment[:, 19]
    segment_set = np.asarray(segment[:, :19].astype(float))
    datasets_array.append(segment_set)
    labels_array.append(segment_labels)
    names_array.append("segment")

    spambase = np.loadtxt("Datasets/spambase.dat", skiprows=62, delimiter=',', dtype=str)
    spambase_labels = spambase[:, 57]
    spambase_set = np.asarray(spambase[:, :57].astype(float))
    datasets_array.append(spambase_set)
    labels_array.append(spambase_labels)
    names_array.append("spambase")

    banana = np.loadtxt("Datasets/banana.dat", skiprows=7, delimiter=',', dtype=str)
    banana_labels = banana[:, 2]
    banana_set = np.asarray(banana[:, :2].astype(float))
    datasets_array.append(banana_set)
    labels_array.append(banana_labels)
    names_array.append("banana")

    titanic = np.loadtxt("Datasets/titanic.dat", skiprows=8, delimiter=',', dtype=str)
    titanic_labels = titanic[:, 3]
    titanic_set = np.asarray(titanic[:, :3].astype(float))
    datasets_array.append(titanic_set)
    labels_array.append(titanic_labels)
    names_array.append("titanic")

    ####################################################################################

    glass = np.loadtxt("Datasets/glass.dat", skiprows=14, delimiter=',', dtype=str)
    glass_labels = glass[:, 9]
    glass_set = np.asarray(glass[:, :9].astype(float))
    datasets_array.append(glass_set)
    labels_array.append(glass_labels)
    names_array.append("glass")

    rand_set, rand_labels = generate_data_2D([[4, 2], [1, 7], [5, 6]], [[0.8, 0.3], [0.3, 0.5], [1.1, 0.7]], 50)
    datasets_array.append(rand_set)
    labels_array.append(rand_labels)
    names_array.append("rand")

    spiral_set, spiral_labels = twospirals(150)
    spiral_set += 15
    datasets_array.append(spiral_set)
    labels_array.append(spiral_labels)
    names_array.append("spiral")

    moons_set, moons_labels = datasets.make_moons(300, .5, .05, random_state)
    moons_set += 1.5
    datasets_array.append(moons_set)
    labels_array.append(moons_labels)
    names_array.append("moons")

    circles_set, circles_labels = datasets.make_circles(300, .5, .05, random_state, .3)
    circles_set += 1.5
    datasets_array.append(circles_set)
    labels_array.append(circles_labels)
    names_array.append("circles")

    return names_array, datasets_array, labels_array
