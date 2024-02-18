import numpy as np
import tensorflow as tf
from cflow.one_dim_transforms import h_dag_dash_extra
from cflow.one_dim_transforms import h_dag, h_dag_dash, h_dag_extra

# Compares against the value obtained using R
def test_h():
    # Define the order of the Bernstein polynomials
    M = 4
    L_START = 1e-04
    R_START = 0.9999

    # Create a sequence from -1 to 2 with a step of 0.5
    sequence = np.arange(-1, 2.1, 0.5)
    t_i_vals = np.repeat(sequence, 3).reshape(-1, 3)
    t_i = tf.Variable(t_i_vals, dtype=tf.float32)

    # Create theta variable
    theta_vals = np.tile(np.array([0., 2., 2., 2., 2.]), (3, 1))
    batch_size = 7
    theta_batch = np.tile(np.expand_dims(theta_vals, 0), (batch_size, 1, 1))
    theta = tf.constant(theta_batch, dtype=tf.float32)

    # Apply h_dag_dash and h_dag_extra functions
    h_dag_dashd = h_dag_dash(t_i, theta)
    # Check that the shape is correct
    assert h_dag_dashd.shape == (7, 3)
    # Check that the values in row 4 are 0.25
    assert np.allclose(h_dag_dashd.numpy()[3], 0.25)
    # Check that the values in the other rows are nan (now we do clamp the values to [0,1] so no nans)
    #assert np.all(np.isnan(h_dag_dashd.numpy()[:3]))
    #assert np.all(np.isnan(h_dag_dashd.numpy()[4:]))

    h_ti = h_dag_extra(t_i, theta, L_START, R_START) 
    expected_h_ti_values = np.array([
        [-1.999440e+00, -1.999440e+00, -1.999440e+00],
        [-9.997400e-01, -9.997400e-01, -9.997400e-01],
        [-3.996407e-05, -3.996407e-05, -3.996407e-05],
        [3.750000e-01, 3.750000e-01, 3.750000e-01],
        [4.000000e-01, 4.000000e-01, 4.000000e-01],
        [4.000000e-01, 4.000000e-01, 4.000000e-01],
        [4.000000e-01, 4.000000e-01, 4.000000e-01]
    ])
    assert h_ti.shape == (7, 3), "h_ti shape mismatch"
    # Check the values of h_ti
    assert np.allclose(h_ti.numpy(), expected_h_ti_values), "h_ti values mismatch"

    expected_h_dash_extra_values = np.array([
        [1.999400e+00, 1.999400e+00, 1.999400e+00],
        [1.999400e+00, 1.999400e+00, 1.999400e+00],
        [1.999400e+00, 1.999400e+00, 1.999400e+00],
        [2.500000e-01, 2.500000e-01, 2.500000e-01],
        [2.000995e-12, 2.000995e-12, 2.000995e-12],
        [2.000995e-12, 2.000995e-12, 2.000995e-12],
        [2.000995e-12, 2.000995e-12, 2.000995e-12]
    ])

    d = h_dag_dash_extra(t_i, theta, L_START, R_START)
    # Check the shape of h_dash_extra
    assert d.shape == (7, 3), "h_dash_extra shape mismatch"
    # Check the values of h_dash_extra
    assert np.allclose(d.numpy(), expected_h_dash_extra_values), "h_dash_extra values mismatch"

