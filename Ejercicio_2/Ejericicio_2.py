import numpy as np

# Función de activación escalón
def funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

# Derivada de la función de activación escalón (aunque en la práctica no se usa, lo incluimos por completitud)
def derivada_funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

class RedNeuronal:
    def __init__(self, tamaño_entrada, tamaño_oculto, tamaño_salida, tasa_aprendizaje=0.2):
        self.tamaño_entrada = tamaño_entrada
        self.tamaño_oculto = tamaño_oculto
        self.tamaño_salida = tamaño_salida
        self.tasa_aprendizaje = tasa_aprendizaje
        
        # Inicializar los pesos con valores aleatorios
        self.pesos_entrada_oculto = np.random.rand(self.tamaño_entrada, self.tamaño_oculto)
        self.pesos_oculto_salida = np.random.rand(self.tamaño_oculto, self.tamaño_salida)
        
    def propagacion_hacia_adelante(self, entradas):
        # Propagación hacia adelante
        self.entrada_oculta = np.dot(entradas, self.pesos_entrada_oculto)
        self.salida_oculta = funcion_escalon(self.entrada_oculta)
        
        self.entrada_final = np.dot(self.salida_oculta, self.pesos_oculto_salida)
        self.salida_final = funcion_escalon(self.entrada_final)
        
        return self.salida_final
    
    def retropropagacion(self, entradas, objetivos, salidas):
        # Cálculo del error
        errores_salida = objetivos - salidas
        errores_ocultos = np.dot(errores_salida, self.pesos_oculto_salida.T) * derivada_funcion_escalon(self.entrada_oculta)
        
        # Actualización de los pesos
        self.pesos_oculto_salida += self.tasa_aprendizaje * np.dot(self.salida_oculta.T, errores_salida)
        self.pesos_entrada_oculto += self.tasa_aprendizaje * np.dot(entradas.T, errores_ocultos)
    
    def entrenar(self, entradas, objetivos):
        salidas = self.propagacion_hacia_adelante(entradas)
        self.retropropagacion(entradas, objetivos, salidas)

# Datos de entrenamiento de ejemplo (XOR)
entradas = np.array([[0, 0],
                     [0, 1],
                     [1, 0],
                     [1, 1]])
objetivos = np.array([[0],
                      [1],
                      [1],
                      [0]])

# Crear la red neuronal
red = RedNeuronal(tamaño_entrada=2, tamaño_oculto=2, tamaño_salida=1)

# Entrenar la red neuronal
for epoca in range(10000):
    red.entrenar(entradas, objetivos)

print(entradas)
print(objetivos)

# Probar la red neuronal
for dato_entrada in entradas:
    print(f"Entrada: {dato_entrada} -> Salida Predicha: {red.propagacion_hacia_adelante(dato_entrada)}")
