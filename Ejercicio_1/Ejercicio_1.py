import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

# Función de activación sigmoidal
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivada de la función de activación sigmoidal
def sigmoid_derivative(x):
    return x * (1 - x)

# Evaluación del modelo
def evaluate_model(X, y_encoded, weights_input_hidden, weights_hidden_output):
    correct_predictions = 0
    total_samples = len(X)

    for i in range(total_samples):
        hidden_input = np.dot(X[i], weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output)
        output = sigmoid(output_input)
        # Selecciona la clase con la mayor probabilidad
        predicted_class = np.argmax(output)
        true_class = np.argmax(y_encoded[i])
        # Compara la predicción con la clase verdadera
        if predicted_class == true_class:
            correct_predictions += 1

    accuracy = correct_predictions / total_samples
    return accuracy

# Carga del dataset Iris
iris = load_iris()
X = iris.data
y = iris.target.reshape(-1, 1)

# Codificación one-hot para las salidas
encoder = OneHotEncoder()
y_encoded = encoder.fit_transform(y).toarray()

# Inicialización de los pesos de manera aleatoria
np.random.seed(1)
input_neurons = 4
hidden_neurons = 5
output_neurons = 3
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
weights_hidden_output = np.random.uniform(size=(hidden_neurons, output_neurons))

# Hiperparámetros
learning_rate = 0.4
epochs = 1000

# Entrenamiento de la red neuronal
tolerance = 1e-5
prev_error = float('inf')
for epoch in range(epochs):
    # Feedforward
    hidden_input = np.dot(X, weights_input_hidden)
    hidden_output = sigmoid(hidden_input)
    output_input = np.dot(hidden_output, weights_hidden_output)
    output = sigmoid(output_input)

    # Backpropagation
    output_error = y_encoded - output
    d_output = output_error * sigmoid_derivative(output)
    
    hidden_error = d_output.dot(weights_hidden_output.T)
    d_hidden = hidden_error * sigmoid_derivative(hidden_output)
    
    # Actualización de pesos
    weights_hidden_output += hidden_output.T.dot(d_output) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate

    # Cálculo del error
    error = np.mean(np.abs(output_error))
    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Error: {error}")
    
    # Verificar convergencia
    if abs(prev_error - error) < tolerance:
        print(f"Convergencia alcanzada en la época {epoch}.")
        break
    
    prev_error = error

print("Entrenamiento completado.")

# Evaluar el modelo después del entrenamiento
accuracy = evaluate_model(X, y_encoded, weights_input_hidden, weights_hidden_output)
print("Precisión del modelo:", accuracy)
