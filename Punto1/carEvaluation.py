from ucimlrepo import fetch_ucirepo 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
  
# Fetch dataset 
car_evaluation = fetch_ucirepo(id=19) 
  
# Data (as pandas dataframes) 
X = car_evaluation.data.features 
y = car_evaluation.data.targets 
  
# Metadata 
print(car_evaluation.metadata) 
  
# Variable information 
print(car_evaluation.variables) 

encoder = OneHotEncoder()

X_encoded = encoder.fit_transform(X).toarray()

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_val, y_train, y_val = train_test_split(X_encoded, y_categorical, test_size=0.2, random_state=42)

model_simple = Sequential()
model_simple.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model_simple.add(Dense(y_categorical.shape[1], activation='softmax'))

model_simple.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_simple = model_simple.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

model_deep = Sequential()
model_deep.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
model_deep.add(Dense(64, activation='relu'))
model_deep.add(Dense(32, activation='relu'))
model_deep.add(Dense(y_categorical.shape[1], activation='softmax'))

model_deep.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history_deep = model_deep.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

simple_eval = model_simple.evaluate(X_val, y_val, verbose=0)
deep_eval = model_deep.evaluate(X_val, y_val, verbose=0)

results = [
    {'Modelo': 'MLP Superficial', 'Precisión': simple_eval[1]},
    {'Modelo': 'MLP Profunda', 'Precisión': deep_eval[1]}
]

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Precisión', ascending=False)

results_df.to_csv('resultados_comparacion.csv', index=False)

print(f'Resultados:\n{results_df}')
