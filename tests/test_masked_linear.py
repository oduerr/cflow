import numpy as np
import tensorflow as tf
from cflow.masked_linear import dag_loss, create_masks_np, create_theta_tilde_maf

def test_dag_loss():
    # Define the order of the Bernstein polynomials
    M = 4
    L_START = 1e-04
    R_START = 0.9999

    # Create a sequence from -1 to 2 with a step of 0.5
    sequence = np.arange(-1, 2.1, 0.5)
    t_i_vals = np.repeat(sequence, 3).reshape(-1, 3)
    t_i = tf.Variable(t_i_vals, dtype=tf.float32)

    # Create theta_tilde variable
    theta_vals = np.tile(np.array([0., 2., 2., 2., 2.]), (3, 1))
    batch_size = 7
    theta_batch = np.tile(np.expand_dims(theta_vals, 0), (batch_size, 1, 1))
    theta = tf.constant(theta_batch, dtype=tf.float32)

    loss = dag_loss(t_i, theta)

    # Check the shape of the loss
    assert loss.shape == (), "Loss shape mismatch"

    # Check the value of the loss
    expected_loss_value = 1.4300749
    assert np.allclose(loss.numpy(), expected_loss_value), "Loss value mismatch"


def test_causal():
    hidden_features = (2,2)
    order = 4
    bs = 5
    dim = 3
    adjacency = np.array([[0, 0, 0],
                         [1, 0, 0],
                         [1, 0, 0]]) == 1 #Needs to be a boolean matrix
    
    masks = create_masks_np(adjacency = adjacency, hidden_features=hidden_features)

    layer_sizes = (adjacency.shape[1], *hidden_features, adjacency.shape[0])
    model = create_theta_tilde_maf(adjacency=adjacency, order = order, layer_sizes=layer_sizes, masks=masks)

    # translated from the R-code x = tf$ones(c(2L,3L))
    x = tf.ones((bs,dim))
    #theta_tilde = param_model(x)
    theta_tilde = model(x)
    assert theta_tilde.shape == (bs, dim, order), "Theta_tilde shape mismatch"

    with tf.GradientTape(persistent = True) as tape:
        tape.watch(x)
        y = model(x)
    d = tape.jacobian(y, x)
    assert d.shape == (bs, dim, order, bs, dim), "Jacobian shape mismatch"
    print(d)
    
 

