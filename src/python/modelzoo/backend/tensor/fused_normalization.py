import tensorflow as tf


def fused_batch_normalization(x, mean, var, beta, gamma, epsilon=1e-3, is_training=True):
    """Applies batch normalization on x given mean, var, beta and gamma.

    I.e. returns:
    `output = (x - mean) / (sqrt(var) + epsilon) * gamma + beta`

    # Arguments
        x: Input tensor or variable.
        mean: Mean of batch.
        var: Variance of batch.
        beta: Tensor with which to center the input.
        gamma: Tensor by which to scale the input.
        epsilon: Fuzz factor.

    # Returns
        A tensor.
    """
    if is_training:
        return tf.nn.fused_batch_norm(x, mean=None, variance=None, offset=beta, scale=gamma, epsilon=epsilon,
                                      is_training=True)
    else:
        return tf.nn.fused_batch_norm(x, mean=mean, variance=var, offset=beta, scale=gamma, epsilon=epsilon,
                                      is_training=False)[0]
