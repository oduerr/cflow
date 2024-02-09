import tensorflow as tf
from tensorflow import keras
from keras import layers, Sequential, Input
import numpy as np

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
        # Ensure the mask is the correct shape and convert it to a tensor
        if self.mask is not None:
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

######### Testing the LinearMasked layer #########
mask = tf.constant([[1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0],
                                 [1, 0, 0]], dtype=tf.float32)
mask = tf.transpose(mask)

model = Sequential()
model.add(LinearMasked(units=4, mask=mask))
#....
x = tf.ones((5, 3))
y = model(x)  # Apply the mask
print(y)

######## Testing the model ########
hidden_features = (2,2)
adjacency = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]]) == 1 #Needs to be a boolean matrix
masks = create_masks_np(adjacency = adjacency, hidden_features=hidden_features)
print(masks)

######### Keras Model #########
layer_sizes = (adjacency.shape[1], *hidden_features, adjacency.shape[0])
model = create_theta_tilde_maf(adjacency=adjacency, order = 2, layer_sizes=layer_sizes, masks=masks)
print(model.summary())
x = tf.ones((5,3))
print(model(x))
d = tf.random.normal((5,3), stddev=3.0)
print(model(d))

from keras.utils import plot_model
plot_model(model, to_file='model_graph.png', show_shapes=True)





