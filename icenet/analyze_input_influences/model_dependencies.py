import tensorflow as tf


############# Stuff needed to load the model #############
@tf.keras.utils.register_keras_serializable()
class TemperatureScale(tf.keras.layers.Layer):
    """
    Implements the temperature scaling layer for probability calibration,
    as introduced in Guo 2017 (http://proceedings.mlr.press/v70/guo17a.html).
    """

    def __init__(self, **kwargs):
        super(TemperatureScale, self).__init__()
        self.temp = tf.Variable(
            initial_value=1.0, trainable=False, dtype=tf.float32, name="temp"
        )

    def call(self, inputs):
        """Divide the input logits by the T value."""
        return tf.divide(inputs, self.temp)

    def get_config(self):
        """For saving and loading networks with this custom layer."""
        return {"temp": self.temp.numpy()}


### Custom Dropout that defaults to applying dropout (convenient for dropout monte carlo)
class DropoutWDefaultTraining(tf.keras.layers.Dropout):
    """Applies Dropout to the input.
    This is a slightly modified version of code from source: Thank you, alxhrzg! https://github.com/keras-team/keras/issues/9412.

    Dropout consists in randomly setting
    a fraction `rate` of input units to 0 at each update during training time,
    which helps prevent overfitting.
    # Arguments
        rate: float between 0 and 1. Fraction of the input units to drop.
        noise_shape: 1D integer tensor representing the shape of the
            binary dropout mask that will be multiplied with the input.
            For instance, if your inputs have shape
            `(batch_size, timesteps, features)` and
            you want the dropout mask to be the same for all timesteps,
            you can use `noise_shape=(batch_size, 1, features)`.
        seed: A Python integer to use as random seed.
        training: bool. If true, dropout is applied.
    # References
        - [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
    """

    def __init__(self, rate, training=True, noise_shape=None, seed=None, **kwargs):
        super().__init__(rate, noise_shape=noise_shape, seed=None, **kwargs)
        self.training = training

    def call(self, inputs, training=None):
        if 0.0 < self.rate < 1.0:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs():
                return tf.keras.backend.dropout(
                    inputs, self.rate, noise_shape, seed=self.seed
                )

            if not training:
                return tf.keras.backend.in_train_phase(
                    dropped_inputs, inputs, training=self.training
                )
            return tf.keras.backend.in_train_phase(
                dropped_inputs, inputs, training=training
            )
        return inputs


class ConstructLeadtimeAccuracy(tf.keras.metrics.CategoricalAccuracy):

    """Computes the network's accuracy over the active grid cell region
    for either a) a specific lead time in months, or b) over all lead times
    at once."""

    def __init__(
        self,
        name="construct_custom_categorical_accuracy",
        use_all_forecast_months=True,
        single_forecast_leadtime_idx=None,
        **kwargs
    ):
        super().__init__(name=name, **kwargs)

        self.use_all_forecast_months = use_all_forecast_months
        self.single_forecast_leadtime_idx = single_forecast_leadtime_idx

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.use_all_forecast_months:
            # Make class dimension final dimension for CategoricalAccuracy
            y_true = tf.transpose(y_true, [0, 1, 2, 4, 3])
            y_pred = tf.transpose(y_pred, [0, 1, 2, 4, 3])
            if sample_weight is not None:
                sample_weight = tf.transpose(sample_weight, [0, 1, 2, 4, 3])

            super().update_state(y_true, y_pred, sample_weight=sample_weight)

        elif not self.use_all_forecast_months:

            super().update_state(
                y_true[..., self.single_forecast_leadtime_idx],
                y_pred[..., self.single_forecast_leadtime_idx],
                sample_weight=sample_weight[..., self.single_forecast_leadtime_idx] > 0,
            )

    def result(self):
        return 100 * super().result()

    def get_config(self):
        """For saving and loading networks with this custom metric."""
        return {
            "single_forecast_leadtime_idx": self.single_forecast_leadtime_idx,
            "use_all_forecast_months": self.use_all_forecast_months,
        }

    @classmethod
    def from_config(cls, config):
        """For saving and loading networks with this custom metric."""
        return cls(**config)


############################################################
