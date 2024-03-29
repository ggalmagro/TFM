import numpy as np
import random
import copy as cp

class ISA:

    def __init__(self, data, ml_const, cl_const, nb_clust, segment_percent, ls_max_neighbors, nu, phi, final_temp):

        self._data = data
        self._ml = ml_const
        self._cl = cl_const
        self._dim = data.shape[0]
        self._result_nb_clust = nb_clust
        self._evals_done = 0
        self._segment_size = int(np.ceil(segment_percent * self._dim))
        self._max_neighbors = ls_max_neighbors
        self._nu = nu
        self._phi = phi
        self._final_temp = final_temp


    def init_ils(self):

        self._best_solution = np.random.randint(0, self._result_nb_clust, self._dim)
        self._best_fitness = self.get_single_fitness(self._best_solution)[0]

    # Funcionalidad para evaluar un unico individuo decodificado
    def get_single_fitness(self, cromosome):

        current_clustering = cromosome
        # Inicializamos la distancia media de las instancias de los clusters
        total_mean_distance = 0
        # Obtenemos el numero de clusters del clustering actual
        nb_clusters = len(set(current_clustering))

        # Para cada cluster en el clustering actual
        for j in set(current_clustering):
            # Obtener las instancias asociadas al cluster
            clust = self._data[current_clustering == j, :]

            if clust.shape[0] > 1:

                # Obtenemos la distancia media intra-cluster
                tot = 0.
                for k in range(clust.shape[0] - 1):
                    tot += ((((clust[k + 1:] - clust[k]) ** 2).sum(1)) ** .5).sum()

                if ((clust.shape[0] - 1) * (clust.shape[0]) / 2.) == 0.0:
                    print(tot)
                    print(cromosome)
                    print(clust)

                avg = tot / ((clust.shape[0] - 1) * (clust.shape[0]) / 2.)

                # Acumular la distancia media
                total_mean_distance += avg

        # Inicializamos el numero de restricciones que no se satisfacen
        infeasability = 0

        # Calculamos el numero de restricciones must-link que no se satisfacen
        for c in range(np.shape(self._ml)[0]):

            if current_clustering[self._ml[c][0]] != current_clustering[self._ml[c][1]]:
                infeasability += 1

        # Calculamos el numero de restricciones cannot-link que no se satisfacen
        for c in range(np.shape(self._cl)[0]):

            if current_clustering[self._cl[c][0]] == current_clustering[self._cl[c][1]]:
                infeasability += 1

        # Calcular el valor de la funcion fitness
        distance = total_mean_distance / nb_clusters
        # penalty = self._mu * self._dim * infeasability
        penalty = distance * infeasability
        fitness = distance + penalty

        # Aumentar en uno el contador de evaluacions de la funcion objetivo
        self._evals_done += 1

        return fitness, distance, penalty

    def segment_mutation_operator(self, chromosome):

        segment_start = np.random.randint(self._dim)
        segment_end = (segment_start + self._segment_size) % self._dim
        new_segment = np.random.randint(0, self._result_nb_clust, self._segment_size)
        if segment_start < segment_end:

            chromosome[segment_start:segment_end] = new_segment

        else:

            chromosome[segment_start:] = new_segment[:self._dim - segment_start]
            #np.random.randint(0, self._result_nb_clust, self._dim - segment_start)
            chromosome[:segment_end] = new_segment[self._dim - segment_start:]
            #np.random.randint(0, self._result_nb_clust, segment_end)

        return chromosome

    def random_mutation_operator(self, chromosome):

        pos = np.random.choice(self._dim, self._segment_size, replace=False)
        new_labels = np.random.randint(0, self._result_nb_clust, self._segment_size)

        chromosome[pos] = new_labels

        return chromosome

    def simulated_annealing(self, chromosome):

        generated = 0
        improvement = True
        random_index_list = np.array(range(self._dim))
        random.shuffle(random_index_list)
        ril_ind = 0
        fitness = self.get_single_fitness(chromosome)[0]

        #Establecemos los parametros del esquema de enfiramiento
        M = self._max_neighbors
        T = (self._nu * fitness) / (-np.log(self._phi))
        beta = (T - self._final_temp)/(self._max_neighbors * T * self._final_temp)

        #while improvement and generated < self._max_neighbors:
        while generated < self._max_neighbors:

            object_index = random_index_list[ril_ind]
            improvement = False
            original_label = chromosome[object_index]
            other_labels = np.delete(np.array(range(self._result_nb_clust)), original_label)
            random.shuffle(other_labels)

            for label in other_labels:

                generated += 1
                chromosome[object_index] = label
                new_fitness = self.get_single_fitness(chromosome)[0]
                improvement = new_fitness - fitness

                if improvement < 0 or np.random.rand() < np.exp(- (improvement / T)):

                    fitness = new_fitness
                    improvement = True
                    break

                else:

                    chromosome[object_index] = original_label

            T /= (1 + beta * T)

            if ril_ind == self._dim - 1:

                random.shuffle(random_index_list)
                ril_ind = 0

            else:

                ril_ind += 1

        return chromosome, fitness


    def run(self, max_evals):

        self.init_ils()

        while self._evals_done < max_evals:

            #print("Evaluaciones: " + str(self._evals_done) + " valor: " + str(self._best_fitness))

            mutant = self.segment_mutation_operator(self._best_solution)
            improved_mutant, improved_mutant_fitness = self.simulated_annealing(mutant)

            if improved_mutant_fitness < self._best_fitness:

                self._best_solution = mutant
                self._best_fitness = improved_mutant_fitness


        return self._best_fitness, self._best_solution














