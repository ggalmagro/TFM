import numpy as np
from functions import *


def main():

    names_array, datasets_array, labels_array = load_all_datasets()
    const_percent_vector = np.array([0.05, 0.1, 0.15, 0.2])
    restrictions_file = open("Results/Constraints/restrictions_file.txt", "w+")

    for label_percent in const_percent_vector:

        restrictions_file.write("------------ Procesando en porcentaje de restricciones: " + str(label_percent) + " ------------\n")
        restrictions_file.write("Dataset TotalRestr   ML(%)   CL(%)\n")

        for i in range(len(names_array)):

            name = names_array[i]
            labels = labels_array[i]
            set_size = len(labels)

            #nb_const = int(((set_size * (set_size - 1))/2) * label_percent)
            nb_const = int(np.ceil(label_percent * set_size) * (np.ceil(label_percent * set_size) - 1)/2)
            print("Procesando " + name + " con " + str(len(labels)) + " etiquetas y " + str(nb_const) + " restricciones")
            const = gen_rand_const(labels, nb_const)
            np.savetxt("Restr/" + str(name) + "(" + str(label_percent) + ").txt", const, fmt='%5d')

            ml_const, cl_const = get_const_list(const)
            ml_const_percent = (len(ml_const) / nb_const) * 100
            cl_const_percent = (len(cl_const) / nb_const) * 100

            restrictions_file.write(name + " & " + str(nb_const) + " & " + str(ml_const_percent) + " & " + str(
                cl_const_percent) + " \\\\ \n")

    restrictions_file.close()


if __name__ == "__main__": main()