# 💧 Teaching a Neural Network Quantum Chemistry

Can a machine *learn* the physics of a molecule?  
In this project, I trained a neural network to predict the **potential energy of water molecules** directly from their atomic coordinates — bridging quantum chemistry and machine learning.

---

## 🧠 Overview

Using **TensorFlow** and **Keras**, I built a model that learns the relationship between 3D molecular geometry and total energy for H₂O.  
The dataset included **rotated**, **unrotated**, and **test** configurations, each read from `.xyz` and `.ENER` files.  
All inputs were scaled and concatenated into a single training set before being converted into tensors for model training.

---

## ⚙️ Model Architecture

- **Input:** 9 atomic coordinate features  
- **Hidden layers:** Two dense layers (64 neurons each, ReLU activation)  
- **Output:** Single scalar energy value (in eV)  
- **Optimizer:** Adam  
- **Loss function:** Mean Squared Error (MSE)  
- **Regularization:** Early stopping with validation split  

---

## 📈 Results

The model achieved **low MAE and MSE** on the test set, showing excellent generalization to unseen geometries.  
Training and validation curves displayed smooth convergence, indicating a stable and well-tuned network.  

This approach demonstrates how **machine learning can replicate physical energy landscapes** — capturing complex quantum behaviour using data-driven methods.

---

## 🔬 Key Takeaways

- Machine learning can model molecular potential energy surfaces efficiently  
- The 9-input neural representation encodes meaningful geometric relationships  
- Even a simple feed-forward network can predict quantum-like trends in energy  

---

📄 **Full Report:**  
[Predicting Potential Energy of Water Molecules Using Neural Networks](https://github.com/Aidan-Doyle-2/Predicting-potential-energy-of-water-using-neural-networks/blob/main/PE%20of%20water%20molecules%20report.pdf)

🎓 *Developed as part of my MSc in Quantum Science & Technology (Trinity College Dublin)*  
