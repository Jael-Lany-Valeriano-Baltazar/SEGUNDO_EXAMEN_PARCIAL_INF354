import numpy as np
import random

grafo = {
    'A': {'C': 9, 'B': 7, 'E': 20, 'D': 8},
    'B': {'A': 7, 'C': 10, 'E': 11, 'D': 4},
    'C': {'A': 9, 'B': 10, 'D': 15, 'E': 5},  
    'D': {'A': 8, 'B': 4, 'C': 15, 'E': 17},
    'E': {'A': 20,'B': 11, 'C': 5, 'D':17}
}

def calcular_distancia(camino):
    distancia_total = 0
    for i in range(len(camino) - 1):
        distancia_total += grafo[camino[i]][camino[i + 1]]
    return distancia_total

def generar_camino_aleatorio():
    nodos_restantes = list(grafo.keys())
    nodos_restantes.remove('A')  # Eliminar 'A' de los nodos restantes
    camino = ['A']  # Comenzar con el nodo 'A'
    camino += random.sample(nodos_restantes, len(nodos_restantes))
    return camino



def seleccion_torneo(poblacion, k=3):
    torneo_idx = np.random.choice(len(poblacion), k, replace=False)
    torneo = [poblacion[i] for i in torneo_idx]
    return min(torneo, key=lambda x: calcular_distancia(x))

def cruzar(padre1, padre2):
    punto_cruce = random.randint(0, len(padre1) - 1)
    hijo = padre1[:punto_cruce]
    for nodo in padre2:
        if nodo not in hijo:
            hijo = np.concatenate((hijo, [nodo]))
    return hijo


def mutar(camino, probabilidad_mutacion):
    if random.random() < probabilidad_mutacion:
        idx1, idx2 = random.sample(range(len(camino)), 2)
        camino[idx1], camino[idx2] = camino[idx2], camino[idx1]

def algoritmo_genetico(num_generaciones, tamano_poblacion, probabilidad_mutacion):
    poblacion = [generar_camino_aleatorio() for _ in range(tamano_poblacion)]
    for _ in range(num_generaciones):
        nueva_generacion = []
        for _ in range(tamano_poblacion):
            padre1 = seleccion_torneo(poblacion)
            padre2 = seleccion_torneo(poblacion)
            hijo = cruzar(padre1, padre2)
            mutar(hijo, probabilidad_mutacion)
            nueva_generacion.append(hijo.tolist())  # Convertir a lista
        poblacion = nueva_generacion
    mejor_camino = min(poblacion, key=lambda x: calcular_distancia(x))
    return mejor_camino, calcular_distancia(mejor_camino)


# Ejemplo de uso
mejor_camino, distancia = algoritmo_genetico(num_generaciones=100, tamano_poblacion=100, probabilidad_mutacion=0.1)
print("Mejor camino encontrado:", mejor_camino)
print("Distancia del mejor camino:", distancia)
