import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Input
import numpy as np

from cflow.one_dim_transforms import h_dag_dash_extra, h_dag_extra

def create_masks_np(adjacency, hidden_features=(64, 64), activation='relu'):
    out_features, in_features = adjacency.shape
    adjacency, inverse_indices = np.unique(adjacency, axis=0, return_inverse=True)
    precedence = np.dot(adjacency.astype(int), adjacency.T.astype(int)) == adjacency.sum(axis=-1, keepdims=True).T
    masks = []
    for i, features in enumerate((*hidden_features, out_features)):
        if i > 0:
            mask = precedence[:, indices]
        else:
            mask = adjacency
        if np.all(~mask):
            raise ValueError("The adjacency matrix leads to a null Jacobian.")

        if i < len(hidden_features):
            reachable = np.nonzero(mask.sum(axis=-1))[0]
            if len(reachable) > 0:
                indices = reachable[np.arange(features) % len(reachable)]
            else:
                indices = np.array([], dtype=int)
            mask = mask[indices]
        else:
            mask = mask[inverse_indices]
        mask = tf.convert_to_tensor(mask, dtype=tf.float32)
        masks.append(mask)
    return masks

class LinearMasked(keras.layers.Layer):
    """
    A linear layer with a mask applied to the weights.
    """
    
    def __init__(self, units=32, mask=None, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.units = units
        self.mask = mask  # Add a mask parameter

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer="random_normal",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(self.units,), initializer="random_normal", trainable=True
        )

        # Handle the mask conversion if it's a dictionary (when loaded from a saved model)
        if self.mask is not None:
            if isinstance(self.mask, dict) or isinstance(self.mask, tf.__internal__.tracking.AutoTrackable):
                # Extract the mask value and dtype from the dictionary
                mask_value = self.mask.get('config').get('value')
                mask_dtype = self.mask.get('config').get('dtype')
                # Convert the mask value back to a numpy array
                mask_np = np.array(mask_value, dtype=mask_dtype)
                # Convert the numpy array to a TensorFlow tensor
                self.mask = tf.convert_to_tensor(mask_np, dtype=mask_dtype)
            else:
                # Ensure the mask is the correct shape and convert it to a tensor
                if self.mask.shape != self.w.shape:
                    raise ValueError("Mask shape must match weights shape")
                self.mask = tf.convert_to_tensor(self.mask, dtype=self.w.dtype)

    def call(self, inputs):
        if self.mask is not None:
            # Apply the mask
            masked_w = self.w * self.mask
        else:
            masked_w = self.w
        return tf.matmul(inputs, masked_w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'mask': self.mask.numpy() if self.mask is not None else None,  # Convert mask to numpy array if it's a tensor
        })
        return config

@tf.keras.utils.register_keras_serializable()
def dag_loss(t_i, theta_tilde):
    """
        The loss function for the MAF model.
        @param t_i: The observed values of shape (batch_size, |x|) where |x| is the number of features.
        @param theta_tilde: The unrestricted parameters of shape (batch_size, |x|, order) usually output by the MAF model. 
    """
    L_START = 0.0001
    R_START = 0.9999
    
    theta = to_theta3(theta_tilde)
    h_ti = h_dag_extra(t_i, theta, L_START, R_START)
    log_latent_density = -h_ti - 2 * tf.math.softplus(-h_ti)
    h_dag_dashd = h_dag_dash_extra(t_i, theta, L_START, R_START)
    log_lik = log_latent_density + tf.math.log(tf.math.abs(h_dag_dashd))
    return -tf.reduce_mean(log_lik)


def create_theta_tilde_maf(adjacency, order, layer_sizes, masks):
    input_layer = Input(shape=(adjacency.shape[1],))
    outs = []
    for r in range(1, order + 1):
        d = input_layer
        for i in range(1, len(layer_sizes) - 1):
            d = LinearMasked(units=layer_sizes[i], mask=1.0*tf.transpose(masks[i - 1]), name=f"LM_o{r}_l{i}")(d)
            d = layers.Activation('relu')(d)
        
        out = LinearMasked(units=layer_sizes[-1], mask=1.0*tf.transpose(masks[-1]), name=f"LM_o{r}_ll")(d)
        outs.append(tf.expand_dims(out, axis=-1)) # (None, |x|, 1) for concatenation later in last layer

    outs_c = layers.Concatenate(axis=-1)(outs) # (None, |x|, order) concatenation on the last axis
    model = keras.models.Model(inputs=input_layer, outputs=outs_c)
    return model

def to_theta3(theta_tilde):
    """
    Converts the unrestricted input parameter theta_tilde to the restricted theta. 
    The input parameter theta_tilde is a tensor of shape (batch_size, |x|, order) 
    where |x| is the number of features and order is the order of the MAF. 

    Parameters:
    theta_tilde (type): Description of parameter theta_tilde.

    Returns:
    type: Description of return value.
    """
    if len(theta_tilde.shape) != 3:
        raise ValueError("theta_tilde must have shape (batch_size, |x|, order)")

    # Ensure that shift is the same dtype as theta_tilde
    shift = tf.convert_to_tensor(np.log(2) * theta_tilde.shape[-1] / 2, dtype=theta_tilde.dtype)
    
    order = tf.shape(theta_tilde)[2]
    widths = tf.nn.softplus(theta_tilde[:, :, 1:order])
    widths = tf.concat([theta_tilde[:, :, 0:1], widths], axis=-1)
    
    # Ensure subtraction happens with tensors of the same dtype
    return tf.cumsum(widths, axis=-1) - shift

# Sample Standard Logistic
def sample_standard_logistic(shape, epsilon=1e-7):
    uniform_samples = tf.random.uniform(shape, minval=0, maxval=1)
    clipped_uniform_samples = tf.clip_by_value(uniform_samples, epsilon, 1 - epsilon)
    logistic_samples = tf.math.log(clipped_uniform_samples / (1 - clipped_uniform_samples))
    return logistic_samples



