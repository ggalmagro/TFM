import numpy as np
import copy as cp
import scipy.stats

class SHADE:

    def __init__(self, data, ml_const, cl_const, population_size, nb_clust, ls = False):

        self._data = data
        self._ml = ml_const
        self._cl = cl_const
        self._dim = data.shape[0]
        self._population_size = population_size
        self._result_nb_clust = nb_clust
        self._external_archive = np.empty((0, 0))
        self._h_record_cr = np.empty((0, 0))
        self._h_record_f = np.empty((0, 0))
        self._ls = ls


    def init_population(self):

        self._external_archive = np.zeros((self._population_size, self._dim))
        self._population = np.random.rand(self._population_size, self._dim)
        self._h_record_cr = np.full(self._population_size, 0.5)
        self._h_record_f = np.full(self._population_size, 0.5)

        self._best = self.decode_random_key(self._population[0, :])
        self._best_fitness = self.get_single_fitness(self._population[0, :])[0]


    # Funcionalidad para decodificar un unico individuo
    def decode_random_key(self, cromosome):

        decoded = np.ceil(cromosome * self._result_nb_clust)
        decoded[decoded == 0] = 1
        return decoded - 1


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


    # Funcionalidad para evaluar un unico individuo decodificado
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
        #penalty = self._mu * self._dim * infeasability
        penalty = distance * infeasability
        fitness = distance + penalty

        # Aumentar en uno el contador de evaluacions de la funcion objetivo
        self._evals_done += 1

        return fitness, distance, penalty

    def run(self, max_eval):

        self._evals_done = 0
        self.init_population()
        generations = 0
        h_index = 0
        external_archive_size = 0

        # Mientras no se haya alcanzado el numero maximo de evaluaciones
        while self._evals_done < max_eval:

            #Ordenamos la poblacion segun el valor fitness de cada individuo
            self._population_fitness = self.get_fitness()[0]
            sorted_population_index = np.argsort(self._population_fitness)
            self._population_fitness = self._population_fitness[sorted_population_index]
            self._population = self._population[sorted_population_index]

            if self._population_fitness[0] < self._best_fitness:

                self._best = cp.deepcopy(self.decode_random_key(self._population[0, :]))
                self._best_fitness = self._population_fitness[0]

            if self._ls:

                self.local_search()

            #print(str(generations) + " " + str(self._best_fitness))

            #Inicializamos estructuras de almacenamiento auxiliares
            s_cr = np.empty((0, 0))
            s_f = np.empty((0, 0))
            fitness_improvements = np.empty((0, 0))
            next_gen = np.empty(self._population.shape)

            for j in range(self._population_size):

                #Obtener valores necearios para la generacion de individuos
                r_i = np.random.randint(0, self._population_size)
                cr_i = np.random.normal(self._h_record_cr[r_i], 0.1)
                f_i = scipy.stats.cauchy.rvs(loc=self._h_record_f[r_i], scale=0.1)
                p_i = np.random.uniform(2.0/self._population_size, 0.2)

                #Obtenemos los individuos que intervienen en la generacion
                x_i = self._population[j]
                x_pbest = self._population[np.random.randint(0, int(self._population_size * p_i))]
                x_r1 = self._population[np.random.randint(0, self._population_size)]

                r2_index = np.random.randint(0, self._population_size + external_archive_size)

                if r2_index < self._population_size:

                    x_r2 = self._population[r2_index]

                else:

                    x_r2 = self._external_archive[r2_index - self._population_size]

                #Generamos el mutante segun el operador current-to-best/1
                mutant = x_i + f_i * (x_pbest - x_i) + f_i * (x_r1 - x_r2)
                mutant = np.clip(mutant, 0, 1)

                #Corregimos los valores fuera el rango [0,1] del mutante
                # for i in range(len(mutant)):
                #
                #     if mutant[i] < 0 or mutant[i] > 1:
                #         mutant[i] = np.random.rand()

                #Obtenemos los puntos de cruce segun cr_i
                cross_points = np.random.rand(self._dim) <= cr_i

                #Obtenemos el vector hijo de x_i
                trial = np.where(cross_points, mutant, x_i)

                #Obtenemos la funcion fitness para el nuevo individuo
                f = self.get_single_fitness(trial)[0]

                #Aplicamos el operador de sustitucion
                if f <= self._population_fitness[j]:

                    next_gen[j, :] = trial

                else:

                    next_gen[j, :] = x_i

                #Actualizamos la informacion para el calculo de los parametros adaptativos
                if f < self._population_fitness[j]:

                    s_cr = np.append(s_cr, cr_i)
                    s_f = np.append(s_f, f_i)
                    fitness_improvements = np.append(fitness_improvements, [self._population_fitness[j] - f])

                    #Si x_i ha sido sustituido por su descendencia, almacenamos x_i en el archivo externo
                    if external_archive_size < len(self._external_archive):

                        self._external_archive[external_archive_size, :] = x_i
                        external_archive_size += 1

                    else:

                        self._external_archive[np.random.randint(0, external_archive_size), :] = x_i

            #Actualizamos los valores de los parametros autoadaptativos
            if len(s_cr) != 0 and len(s_f) != 0:

                #Calculamos los pesos ponderados
                w_k = fitness_improvements / fitness_improvements.sum()

                #Calculamos la media ponderada de S_Cr para actualizar H
                mean_wa = (w_k * s_cr).sum()
                self._h_record_cr[h_index] = mean_wa

                #Calculamos la media ponderada de Lehmer de S_F para actualizar H
                mean_wl = (w_k * (s_f**2)).sum() / (w_k * s_f).sum()
                self._h_record_f[h_index] = mean_wl

                h_index = (h_index + 1) % self._population_size

            #Sustituimos la poblacion por la nueva generacion
            self._population = cp.deepcopy(next_gen)
            generations += 1

        #Obtenemos el mejor individuo tras la iteracion final
        self._population_fitness = self.get_fitness()[0]
        best_index = np.argmin(self._population_fitness)
        self._best_fitness = self._population_fitness[best_index]
        self._best = self.decode_random_key(self._population[best_index])

        return self._best_fitness, self._best

    # Busqueda local por trayectorias simples
    def local_search(self):

        for clust in range(self._population_size):

            current_clustering = cp.deepcopy(self.decode_random_key(self._population[clust, :]))
            current_fitness = self._population_fitness[clust]

            improvement = True
            object_index = 0

            while improvement and object_index < current_clustering.shape[0]:

                improvement = False
                original_label = current_clustering[object_index]
                other_labels = set(range(self._result_nb_clust))
                other_labels.remove(original_label)

                for label in other_labels:

                    current_clustering[object_index] = label
                    new_fitness = self.get_single_fitness(current_clustering)[0]

                    if new_fitness < current_fitness:

                        current_fitness = new_fitness
                        improvement = True

                    else:

                        current_clustering[object_index] = original_label

                object_index += 1

            if current_fitness < self._best_fitness:
                self._best = cp.deepcopy(current_clustering)
                self._best_fitness = current_fitness
