import numpy as np

# Función objetivo (puede ser cualquier función que queramos maximizar)
def objetivo(perm):
    # Ejemplo de función objetivo: la suma de los cuadrados de los elementos de la permutación
    return -np.sum(perm**2)

def vecino(perm):
    # Genera un vecino cambiando dos elementos aleatoriamente
    new_perm = perm.copy()
    idx1, idx2 = np.random.choice(len(perm), 2, replace=False)
    new_perm[idx1], new_perm[idx2] = new_perm[idx2], new_perm[idx1]
    return new_perm

def busqueda_local(N, max_iter):
    # Genera una permutación inicial aleatoria
    mejor_perm = np.random.permutation(N)
    mejor_valor = objetivo(mejor_perm)
    
    for _ in range(max_iter):
        # Genera un vecino y calcula su valor objetivo
        vecino_perm = vecino(mejor_perm)
        vecino_valor = objetivo(vecino_perm)
        
        # Si el vecino es mejor, actualiza la mejor solución
        if vecino_valor > mejor_valor:
            mejor_perm = vecino_perm
            mejor_valor = vecino_valor
    
    return mejor_perm, mejor_valor

# Configuración del problema
N = 10  # Número de elementos
max_iter = 1000  # Número máximo de iteraciones

# Ejecución de la búsqueda local
mejor_perm, mejor_valor = busqueda_local(N, max_iter)

print("Mejor permutación encontrada:", mejor_perm)
print("Valor objetivo de la mejor permutación:", mejor_valor)
