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
train = dgp(50)
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

optimizer = Adam(lr=0.0001)
param_model.compile(optimizer=optimizer, loss=dag_loss)
param_model.summary()

# Prepare the dataset for training
train_data = tf.data.Dataset.from_tensor_slices((train['df_scaled'], train['df_scaled']))
train_data = train_data.batch(32)

# Custom training loop
epochs = 5
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset
    for step, (x_batch_train, y_batch_train) in enumerate(train_data):
        with tf.GradientTape() as tape:
            # Run the forward pass of the layer
            logits = param_model(x_batch_train, training=True)
            
            # Compute the loss value for this minibatch
            loss_value = dag_loss(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss
        grads = tape.gradient(loss_value, param_model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss
        optimizer.apply_gradients(zip(grads, param_model.trainable_weights))

        # Log every 10 batches
        if step % 10 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * 32))

# Fit the model
history = param_model.fit(x=tf.constant(train['df_scaled'], dtype=tf.float32), 
                          y=tf.constant(train['df_scaled'], dtype=tf.float32), 
                          epochs=500, 
                          verbose=True)





# Plot training history
plt.plot(history.epoch, history.history['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
print