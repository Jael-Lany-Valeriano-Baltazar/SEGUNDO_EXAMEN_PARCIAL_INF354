'''import numpy as np

# Función objetivo (número total de conflictos de horario)
def objetivo(asignacion):
    # Aquí calcularíamos el número total de conflictos de horario
    # Supongamos que devolvemos un valor aleatorio para propósitos de demostración
    return np.random.randint(0, 100)

# Genera un vecino cambiando una asignación aleatoria
def vecino(asignacion):
    new_asignacion = asignacion.copy()
    # Aquí implementaríamos el cambio de una asignación aleatoria
    return new_asignacion

# Algoritmo de recocido simulado
def recocido_simulado(N, max_iter, temperatura_inicial, enfriamiento):
    mejor_asignacion = np.random.randint(0, 2, size=N)  # Genera una asignación inicial aleatoria
    mejor_valor = objetivo(mejor_asignacion)
    temperatura = temperatura_inicial

    for _ in range(max_iter):
        nueva_asignacion = vecino(mejor_asignacion)
        nuevo_valor = objetivo(nueva_asignacion)
        delta_valor = nuevo_valor - mejor_valor

        # Si la nueva solución es mejor, acéptala
        if delta_valor < 0 or np.random.rand() < np.exp(-delta_valor / temperatura):
            mejor_asignacion = nueva_asignacion
            mejor_valor = nuevo_valor

        # Reduce la temperatura
        temperatura *= enfriamiento

    return mejor_asignacion, mejor_valor

# Configuración del problema
N = 10  # Número de clases
max_iter = 1000  # Número máximo de iteraciones
temperatura_inicial = 100.0  # Temperatura inicial
enfriamiento = 0.95  # Factor de enfriamiento

# Ejecución del recocido simulado
mejor_asignacion, mejor_valor = recocido_simulado(N, max_iter, temperatura_inicial, enfriamiento)

print("Mejor asignación encontrada:", mejor_asignacion)
print("Valor objetivo de la mejor asignación:", mejor_valor)
'''


import numpy as np
import matplotlib.pyplot as plt

# Función para calcular la distancia entre dos ciudades
def distancia(ciudad1, ciudad2):
    return np.linalg.norm(ciudad1 - ciudad2)

# Función objetivo (longitud total del recorrido)
def objetivo(recorrido, distancias):
    total_distancia = 0
    for i in range(len(recorrido) - 1):
        total_distancia += distancias[recorrido[i], recorrido[i+1]]
    total_distancia += distancias[recorrido[-1], recorrido[0]]  # Regresar al punto de partida
    return total_distancia

# Genera un vecino intercambiando dos ciudades aleatorias
def vecino(recorrido):
    new_recorrido = recorrido.copy()
    idx1, idx2 = np.random.choice(len(recorrido), 2, replace=False)
    new_recorrido[idx1], new_recorrido[idx2] = new_recorrido[idx2], new_recorrido[idx1]
    return new_recorrido

# Algoritmo de recocido simulado
def recocido_simulado(ciudades, max_iter, temperatura_inicial, enfriamiento):
    n_ciudades = len(ciudades)
    distancias = np.zeros((n_ciudades, n_ciudades))
    for i in range(n_ciudades):
        for j in range(n_ciudades):
            distancias[i, j] = distancia(ciudades[i], ciudades[j])

    mejor_recorrido = np.random.permutation(n_ciudades)
    mejor_valor = objetivo(mejor_recorrido, distancias)
    temperatura = temperatura_inicial

    for _ in range(max_iter):
        nuevo_recorrido = vecino(mejor_recorrido)
        nuevo_valor = objetivo(nuevo_recorrido, distancias)
        delta_valor = nuevo_valor - mejor_valor

        # Si la nueva solución es mejor, acéptala
        if delta_valor < 0 or np.random.rand() < np.exp(-delta_valor / temperatura):
            mejor_recorrido = nuevo_recorrido
            mejor_valor = nuevo_valor

        # Reduce la temperatura
        temperatura *= enfriamiento

    return mejor_recorrido, mejor_valor

# Genera ciudades aleatorias en un espacio 2D
np.random.seed(0)  # Para reproducibilidad
n_ciudades = 20
ciudades = np.random.rand(n_ciudades, 2)  # Coordenadas (x, y)

# Configuración del recocido simulado
max_iter = 10000
temperatura_inicial = 100.0
enfriamiento = 0.95

# Ejecución del recocido simulado
mejor_recorrido, mejor_valor = recocido_simulado(ciudades, max_iter, temperatura_inicial, enfriamiento)

# Visualización del resultado
plt.figure(figsize=(8, 6))
plt.scatter(ciudades[:, 0], ciudades[:, 1], color='blue')
for i in range(len(mejor_recorrido) - 1):
    plt.plot([ciudades[mejor_recorrido[i], 0], ciudades[mejor_recorrido[i+1], 0]],
             [ciudades[mejor_recorrido[i], 1], ciudades[mejor_recorrido[i+1], 1]], color='red')
plt.plot([ciudades[mejor_recorrido[-1], 0], ciudades[mejor_recorrido[0], 0]],
         [ciudades[mejor_recorrido[-1], 1], ciudades[mejor_recorrido[0], 1]], color='red')
plt.title(f"Mejor recorrido encontrado (Longitud: {mejor_valor:.2f})")
plt.xlabel("X")
plt.ylabel("Y")
plt.grid(True)
plt.show()
