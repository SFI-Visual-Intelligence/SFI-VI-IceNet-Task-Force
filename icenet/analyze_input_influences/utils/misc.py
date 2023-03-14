import tensorflow as tf


def unit_normalize(x):
    """
    Maps the values in x to the range [0, 1].
    """
    return (x - x.min()) / (x.max() - x.min())


def weight_hessian(model, weights, x, y, loss_fn):
    """
    Calculates the hessian of the weights of a model.
    The hessian is calculated using the jacobian of the gradients of the loss function wrt. the weights.
    The hessian is calculated for each sample in x and then added.
    Note that the calculation is done for each sample in x
    in a loop because tensorflow crashes when trying to calculate the hessian for all samples
    at once.

    Args:
        model (tf.keras.Model): The model to analyze.
        weights (tf.Variable): The weights of the model to calculate the hessian for.
        x (tf.Tensor): Input data.
        y (tf.Tensor): Target data.
        loss_fn (tf.keras.losses): The loss function to use.

    Returns:
        hess (tf.Tensor): The hessian of the weights.
    """
    loss = 0
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            tape1.watch(model.trainable_variables[-2])
            for i in range(x.shape[0]):
                predictions = model(x[i : i + 1])
                loss += loss_fn(y[i : i + 1], predictions)
        grads = tape1.gradient(loss, model.trainable_variables[-2])
    hess = tape2.jacobian(grads, model.trainable_variables[-2])
    return hess
