import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Crear el modelo con las capas especificadas
model = models.Sequential([
    layers.InputLayer(input_shape=(7,)),                    # Capa de entrada con 7 neuronas
    layers.Dense(6, activation='linear'),                   # Primera capa densa (lineal)
    layers.Dense(4, activation='tanh'),                     # Segunda capa densa (tangente-sigmoidal)
    layers.Dense(5, activation='sigmoid'),                  # Tercera capa densa (sigmoidal)
    layers.Dense(4, activation='tanh'),                     # Cuarta capa densa (tangente-sigmoidal)
    layers.Dense(6, activation='linear'),                   # Quinta capa densa (lineal)
    layers.Dense(6, activation=lambda x: x**3)              # Sexta capa densa (cúbica)
])

# Compilar el modelo con un optimizador y función de pérdida (usaremos MSE por simplicidad)
model.compile(optimizer='sgd', loss='mean_squared_error')

# Generar datos de entrada y salida dummy (aleatorios) para entrenamiento
X = np.random.rand(1, 7)  # Una muestra de entrada con 7 características
y = np.random.rand(1, 6)  # Salida esperada con 6 valores (por la capa de salida)

# Realizar una pasada hacia adelante para obtener predicciones (inicialización del modelo)
predictions = model.predict(X)

# Realizar la retropropagación para obtener los gradientes
with tf.GradientTape() as tape:
    predictions = model(X, training=True)  # Paso hacia adelante
    loss = tf.reduce_mean(tf.square(predictions - y))  # Calcular la pérdida (error cuadrático medio)

# Obtener los gradientes de los pesos con respecto a la pérdida
gradients = tape.gradient(loss, model.trainable_weights)

# Mostrar los gradientes de los pesos entre la primera y segunda capa
# Los pesos de la primera capa están en `model.layers[1].kernel`
print("Gradientes de los pesos entre la primera y segunda capa:")
print(gradients[0])  # Gradientes de los pesos de la primera capa

# Mostrar el gradiente específico del peso w(4,2) (que sería la fila 4, columna 2 del primer tensor de pesos)
w42_gradient = gradients[0][3, 1]  # Índices en Python comienzan desde 0
print(f"Gradiente del peso w(4,2): {w42_gradient.numpy()}")
