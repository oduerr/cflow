import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Input
from sklearn.preprocessing import StandardScaler
from cflow.masked_linear import create_masks_np, LinearMasked, create_theta_tilde_maf, dag_loss

######### DGP #########
def dgp(n_obs):
    print("=== Using the DGP of the VACA1 paper in the linear Fashion (Tables 5/6)")
    flip = np.random.choice([0, 1], n_obs)
    X_1 = flip * np.random.normal(-2, np.sqrt(1.5), n_obs) + (1 - flip) * np.random.normal(1.5, 1, n_obs)
    X_2 = -X_1 + np.random.normal(size=n_obs)
    X_3 = X_1 + 0.25 * X_2 + np.random.normal(size=n_obs)

    dat_s = np.column_stack((X_1, X_2, X_3))
    dat_tf = tf.constant(dat_s, dtype=tf.float32)

    # Scaling the data
    scaler = StandardScaler()
    scaled = scaler.fit_transform(dat_tf.numpy()) * 0.99 + 0.005

    A = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    return {'df_orig': dat_tf, 'df_scaled': scaled, 'A': A}

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

M = 30
# Generate training data
train = dgp(500)
# Give summary statistics of df_scaled
print(f"Scaled Training Data:\n Min {train['df_scaled'].min()} Max {train['df_scaled'].max()} Mean {train['df_scaled'].mean()}")


# Transpose the adjacency matrix
adjacency = np.transpose(train['A']) == 1 #Needs to be a boolean matrix
# Define layer sizes
hidden_features = (2,2)
layer_sizes = (adjacency.shape[1], *hidden_features, adjacency.shape[0])


masks = create_masks_np(adjacency = adjacency, hidden_features=hidden_features)
param_model = create_theta_tilde_maf(adjacency=adjacency, order=M+1, layer_sizes=layer_sizes, masks=masks)
from keras.utils import plot_model
plot_model(param_model, to_file='model_graph.png', show_shapes=True)

param_model(tf.constant(train['df_scaled'], dtype=tf.float32))  # Assuming train['df_scaled'] is a numpy array

optimizer = Adam()
param_model.compile(optimizer=optimizer, loss=dag_loss)
param_model.summary()

# Evaluate the model
eval_result = param_model.evaluate(x=tf.constant(train['df_scaled'], dtype=tf.float32), 
                                   y=tf.constant(train['df_scaled'], dtype=tf.float32), 
                                   batch_size=32)

# Fit the model
history = param_model.fit(x=tf.constant(train['df_scaled'], dtype=tf.float32), 
                          y=tf.constant(train['df_scaled'], dtype=tf.float32), 
                          epochs=5, 
                          verbose=True)

# Plot training history
#plt.plot(history.epoch, history.history['loss'])
#plt.xlabel('Epoch')
#plt.ylabel('Loss')
#plt.show()

# Save the model
param_model.save('triangle_test.keras')

# Load the model
#loaded_model = tf.keras.models.load_model('triangle_test.keras', custom_objects={'dag_loss': dag_loss})
loaded_model = tf.keras.models.load_model(
    'triangle_test.keras', 
    custom_objects={
        'LinearMasked': LinearMasked,
        'custom_loss': dag_loss
    }
)


# Evaluate the loaded model
eval_result_loaded = loaded_model.evaluate(x=tf.constant(train['df_scaled'], dtype=tf.float32), 
                                           y=tf.constant(train['df_scaled'], dtype=tf.float32), 
                                           batch_size=32)
print(f"Loaded model loss: {eval_result_loaded}")

eval_result = param_model.evaluate(x=tf.constant(train['df_scaled'], dtype=tf.float32), 
                                   y=tf.constant(train['df_scaled'], dtype=tf.float32), 
                                   batch_size=32)
print(f"Original model loss: {eval_result}")