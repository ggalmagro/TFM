import numpy as np
from sklearn.neighbors.nearest_centroid import NearestCentroid
import copy as cp

class BRKGA:

    def __init__(self, data, population_size, prt_elite, pbt_mutation, pbt_inherit, nb_clust):

        self._data = data
        self._dim = data.shape[0]
        self._population_size = population_size
        self._prt_elite = prt_elite
        self._pbt_mutation = pbt_mutation
        self._pbt_inherit = pbt_inherit
        self._result_nb_clust = nb_clust
        self._evals_done = 0

    #Funcionalidad para inicializar la poblacion de manera aleatoria
    def init_population(self):

        self._population = np.random.rand(self._population_size, self._dim)

    #Funcionalidad para decodificar la poblacion de individuos
    def decode_random_keys(self):

        return np.ceil(self._population * self._result_nb_clust) - 1

    # Funcionalidad para decodificar un unico individuo
    def decode_single_random_key(self, cromosome):

        return np.ceil(cromosome * self._result_nb_clust) - 1

    #Funcionalidad para evaluar la poblacion actual
    def get_fitness(self):

        #Decodificamos la poblacion
        decoded_population = self.decode_random_keys()

        clf = NearestCentroid()
        distances = np.array([])

        #Para cada miembro de la poblacion decodificada
        for i in range(decoded_population.shape[0]):

            #Extraemos el inidviduo a analizar de la poblacion
            current_clustering = decoded_population[i, :]
            #Obtenemos los centroides asociados al clustering descrito por el individuo
            centroids = clf.fit(self._data, current_clustering).centroids_
            #Inicializamos la distancia media de las instancias a su centroide
            total_mean_distance = 0

            #Para cada cluster
            for j in range(self._result_nb_clust):

                #Obtener las instancias asociadas al cluster
                clust = self._data[np.where(current_clustering == j)]
                #Calculamos la distancia euclidea de cada instancia al centroide
                dist = np.linalg.norm(clust - centroids[j, :])
                #Aumentar en uno el contador de evaluacions de la funcion objetivo
                self._evals_done += 1
                #Acumular la distancia media
                total_mean_distance += dist

            #Almacenar la distancia media del clustering analizado
            distances = np.append(distances, total_mean_distance / self._result_nb_clust)

        return distances

    #Operador de cruce aleatorio
    def random_crossover_operator(self, parent1, parent2):

        #Obtenemos el vector de probabilidades de herdar de parent1 y resolvemos las probabilidades
        v = np.where(np.random.rand(self._dim) > self._pbt_inherit)[0]

        #Creamos el nuevo cromosoma como una copia de parent1
        new_cromosome = cp.deepcopy(parent1)

        #Copiamos los genes de parent2 indicados por las probabilidades obtenidas
        new_cromosome[v] = parent2[v]

        return new_cromosome

    def uniform_crossover_operator(self, parent1, parent2):

        #Decodificamos los padres
        decoded_p1 = self.decode_single_random_key(parent1)
        decoded_p2 = self.decode_single_random_key(parent2)

        #Obtenemos las posiciones conicidesntes y no coincidentes de ambos padres
        matches = np.where(decoded_p1 == decoded_p2)
        non_matches = np.where(decoded_p1 != decoded_p2)

        #El nuevo individuo hereda las posiciones coincidentes (calculando la media)
        new_cromosome = self.random_crossover_operator(parent1, parent2)
        new_cromosome[matches] = (parent1[matches] + parent2[matches])/2

        return new_cromosome

    def get_offspring(self, elite, non_elite, offspring_size):

        #Obtenemos listas de indices aleatorios asociados a cromosomas elite y no-elite
        elite_cromosomes_index = np.random.randint(elite.shape[0], size=offspring_size)
        non_elite_cromosomes_index = np.random.randint(non_elite.shape[0], size=offspring_size)

        #Inicializamos la descendencia vacia
        offspring = np.empty((offspring_size, self._dim))

        #Generamos los nuevos inidividuos
        for i in range(offspring_size):

            #Obtenemos cada nuevo inidividuo como un cruce entre un cromosoma elitista y
            #uno no elitista
            new_cromosome = self.uniform_crossover_operator(elite[elite_cromosomes_index[i], :],
                                                    non_elite[non_elite_cromosomes_index[i], :])

            #Almacenamos el nuevo individuo
            offspring[i, :] = new_cromosome

        return offspring

    #Cuerpo principal del AG
    def run(self, max_gen):

        #Inicializamos la poblacion y los parametros necesarios
        self._evals_done = 0
        self.init_population()
        distances = self.get_fitness()
        sorted_dist = np.argsort(distances)
        self._population = self._population[sorted_dist, :]
        num_elite = int(self._population_size * self._prt_elite)
        num_mutants = int(self._population_size * self._pbt_mutation)
        offspring_size = self._population_size - num_elite - num_mutants
        num_generations = 0

        #Mientras no se haya alcanzado el numero maximo de evaluaciones
        while num_generations < max_gen:

            #Guardar la elite de la generacion actual
            elite = self._population[:num_elite, :]
            non_elite = self._population[num_elite:, :]

            #Generar los mutantes de la nueva generacion
            mutants = np.random.rand(num_mutants, self._dim)

            #Generar los descendientes de la nueva generacion cruzando los miembros de la elite
            #con el resto de individuos
            offspring = self.get_offspring(elite, non_elite, offspring_size)

            #Introducimos los nuevos individuos en la poblacion conservando la elite
            self._population[num_elite:, :] = np.vstack((offspring, mutants))

            #Se evalua y reordena la poblacion
            distances = self.get_fitness()
            sorted_dist = np.argsort(distances)
            self._population = self._population[sorted_dist, :]

            num_generations += 1

        return self.decode_single_random_key(self._population[sorted_dist[0]])



