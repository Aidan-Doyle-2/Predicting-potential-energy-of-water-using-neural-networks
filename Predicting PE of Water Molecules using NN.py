import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

print("TensorFlow version:", tf.__version__)

os.chdir("C:/Users/Aidan/Desktop/VScode/ML stuff")

def read_blocks(file_path, num_blocks):
    data = []
    with open(file_path, 'r') as f:
        lines = f.readlines()

        block_size = 5
        for i in range(num_blocks):
            start = i * block_size + 2  
            block_data = []

            for j in range(3):  
                parts = lines[start + j].split()
                block_data.extend(map(float, parts[1:])) 

            data.append(block_data)

    return np.array(data)


def reading_energies(file_path):

    energies = []
    with open(file_path, 'r') as f:
        for line in f:
            energies.append(float(line))
    return np.array(energies)

xyz_unrotated = read_blocks("H2O_unrotated.xyz", 1750)
ener_unrotated = reading_energies("H2O_unrotated.ENER")

xyz_rotated = read_blocks("H2O_rotated.xyz", 1750)
ener_rotated = reading_energies("H2O_rotated.ENER")

xyz_test = read_blocks("H2O_test.xyz", 250)
ener_test = reading_energies("H2O_test.ENER")

xyz_combined_training_data = np.concatenate([xyz_unrotated, xyz_rotated])
ener_combined_training_data = np.concatenate([ener_unrotated, ener_rotated])

scaler = StandardScaler()
scaled_training_xyz = scaler.fit_transform(xyz_combined_training_data)
scaled_test_xyz = scaler.transform(xyz_test)

xyz_tensor_training_data = tf.convert_to_tensor(scaled_training_xyz, dtype=tf.float32)
ener_tensor_training_data = tf.convert_to_tensor(ener_combined_training_data, dtype=tf.float32)

xyz_tensor_test_data = tf.convert_to_tensor(scaled_test_xyz, dtype=tf.float32)
ener_tensor_test_data = tf.convert_to_tensor(ener_test, dtype=tf.float32)

model = models.Sequential([
    
    layers.Input(shape=(9,)), layers.Dense(64, activation='relu'), layers.Dense(64, activation='relu'), layers.Dense(1)                        
    
])


compiled_model = model.compile(
    
    optimizer = 'adam', loss = 'mean_squared_error', metrics = ['mean_absolute_error']

)

history = model.fit(

    xyz_tensor_training_data, ener_tensor_training_data, validation_split=0.2, epochs=100, batch_size=32,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]

)

test_loss, test_mean_absolute_error = model.evaluate(xyz_tensor_test_data, ener_tensor_test_data)
print(f"\nTest Mean Absolute Error: {test_mean_absolute_error:.4f}")
print(f"Test Mean Square Error: {test_loss:.4f}")

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel("Epochs")
plt.ylabel("Mean Square Error")
plt.title("Training Model")
plt.legend()
plt.show()