import tensorflow as tf

def bernstein_basis(tensor, M):
    tensor = tf.convert_to_tensor(tensor)
    dtype = tensor.dtype
    tensor_expanded = tf.expand_dims(tensor, -1)
    # Some are outside the range [0,1] but that should be no problem since they are masked out in h_dag_extra
    # Check if tensor_expanded has values outside [0,1]
    #outside_range = tf.reduce_any((tensor_expanded < 0) | (tensor_expanded > 1))
    #if outside_range:
    #    tf.print("Values outside [0,1] in tensor_expanded:", tensor_expanded)
    #    tf.print("Clipping tensor_expanded to ", tf.keras.backend.epsilon(), "and", 1 - tf.keras.backend.epsilon())
    # Ensuring tensor_expanded is within the range (0,1)
    tensor_expanded = tf.clip_by_value(tensor_expanded, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    

    M = tf.cast(M, dtype)
    k_values = tf.range(M + 1)

    log_binomial_coeff = tf.math.lgamma(M + 1.) - tf.math.lgamma(k_values + 1.) - tf.math.lgamma(M - k_values + 1.)
    log_powers = k_values * tf.math.log(tensor_expanded) + (M - k_values) * tf.math.log(1 - tensor_expanded)
    log_bernstein = log_binomial_coeff + log_powers

    return tf.exp(log_bernstein)

def h_dag(t_i, theta):
    len_theta = tf.shape(theta)[2]
    Be = bernstein_basis(t_i, len_theta - 1)
    return tf.reduce_mean(theta * Be, axis=-1)

def h_dag_dash(t_i, theta):
    len_theta = tf.shape(theta)[2]
    Bed = bernstein_basis(t_i, len_theta - 2)
    dtheta = theta[:, :, 1:len_theta] - theta[:, :, :len_theta - 1]
    return tf.reduce_sum(dtheta * Bed, axis=-1)

def h_dag_extra(t_i, theta, L_START, R_START):
    t_i3 = tf.expand_dims(t_i, axis=-1)
    b0 = tf.expand_dims(h_dag(L_START, theta), axis=-1)
    slope0 = tf.expand_dims(h_dag_dash(L_START, theta), axis=-1)
    mask0 = tf.math.less(t_i3, L_START)
    h = tf.where(mask0, slope0 * (t_i3 - L_START) + b0, t_i3)

    b1 = tf.expand_dims(h_dag(R_START, theta), axis=-1)
    slope1 = tf.expand_dims(h_dag_dash(R_START, theta), axis=-1)
    mask1 = tf.math.greater(t_i3, R_START)
    h = tf.where(mask1, slope1 * (t_i3 - R_START) + b1, h)

    mask = tf.math.logical_and(tf.math.greater_equal(t_i3, L_START), tf.math.less_equal(t_i3, R_START))
    h = tf.where(mask, tf.expand_dims(h_dag(t_i, theta), axis=-1), h)
    return tf.squeeze(h)


def h_dag_dash_extra(t_i, theta, L_START, R_START):
    t_i3 = tf.expand_dims(t_i, axis=-1)
    slope0 = tf.expand_dims(h_dag_dash(L_START, theta), axis=-1)
    mask0 = tf.math.less(t_i3, L_START)
    h_dash = tf.where(mask0, slope0, t_i3)

    slope1 = tf.expand_dims(h_dag_dash(R_START, theta), axis=-1)
    mask1 = tf.math.greater(t_i3, R_START)
    h_dash = tf.where(mask1, slope1, h_dash)

    mask = tf.math.logical_and(tf.math.greater_equal(t_i3, L_START), tf.math.less_equal(t_i3, R_START))
    h_dash = tf.where(mask, tf.expand_dims(h_dag_dash(t_i, theta), axis=-1), h_dash)
    return tf.squeeze(h_dash)