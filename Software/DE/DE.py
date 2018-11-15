import numpy as np
import copy as cp


class DE:

    def __init__(self, data, ml_const, cl_const, population_size, mut, crossp, nb_clust, mu):

        self._data = data
        self._ml = ml_const
        self._cl = cl_const
        self._dim = data.shape[0]
        self._mut = mut
        self._crossp = crossp
        self._population_size = population_size
        self._result_nb_clust = nb_clust
        self._mu = mu
        self._evals_done = 0


    def init_population(self):

        self._population = np.random.rand(self._population_size, self._dim)


    # Funcionalidad para decodificar un unico individuo
    def decode_random_key(self, cromosome):

        decoded = np.ceil(cromosome * self._result_nb_clust)
        decoded[decoded == 0] = 1

        return decoded - 1

    def get_fitness2(self):

        suma = np.sum(self._population, 1)
        self._evals_done += self._population_size

        return suma, suma, suma

    # Funcionalidad para evaluar la poblacion actual
    def get_fitness(self):

        fitness = np.array([])
        distances = np.array([])
        penaltys = np.array([])

        for i in range(self._population_size):

            aux_fitness, aux_dist, aux_penalty = self.get_single_fitness(self._population[i, :])
            fitness = np.append(fitness, aux_fitness)
            distances = np.append(distances, aux_dist)
            penaltys = np.append(penaltys, aux_penalty)

        return fitness, distances, penaltys

    def get_single_fitness2(self, cromosome):

        self._evals_done += 1
        return np.sum(cromosome)


    #Funcionalidad para evaluar un unico individuo decodificado
    def get_single_fitness(self, cromosome):

        # Decodificamos el cromosoma
        current_clustering = self.decode_random_key(cromosome)
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
        penalty = self._mu * self._dim * infeasability
        fitness = distance + penalty

        # if nb_clusters != self._result_nb_clust:
        #     fitness = 99999999999999999999999999999999

        # Aumentar en uno el contador de evaluacions de la funcion objetivo
        self._evals_done += 1

        return fitness, distance, penalty


    def run(self, max_eval):

        self._evals_done = 0
        self.init_population()
        fitness = self.get_fitness()[0]
        best_cromosome_idx = np.argmin(fitness)
        self._best = cp.deepcopy(self.decode_random_key(self._population[best_cromosome_idx, :]))
        self._best_fitness = fitness[best_cromosome_idx]
        generations = 0

        # Mientras no se haya alcanzado el numero maximo de evaluaciones
        while self._evals_done < max_eval:

            for j in range(self._population_size):

                idxs = [idx for idx in range(self._population_size) if idx != j]

                a, b, c = self._population[np.random.choice(idxs, 3, replace=False)]

                mutant = a + self._mut * (b - c)

                for i in range(len(mutant)):

                    if mutant[i] < 0 or mutant[i] > 1:
                        mutant[i] = np.random.rand()

                cross_points = np.random.rand(self._dim) < self._crossp

                if not np.any(cross_points):

                    cross_points[np.random.randint(0, self._dim)] = True

                trial = np.where(cross_points, mutant, self._population[j])

                f = self.get_single_fitness(trial)[0]

                if f < fitness[j]:

                    fitness[j] = f
                    self._population[j] = trial

                    if f < self._best_fitness:

                        self._best_fitness = f
                        self._best = self.decode_random_key(trial)

            generations += 1

            #print(str(generations) + " " + str(self._best_fitness))

        return self._best_fitness, self._best
