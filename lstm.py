import tensorflow as tf
from tensorflow.keras import layers, models

class LSTMTimeSeriesModel(tf.keras.Model):
    """A time series model using an LSTM layer in TensorFlow 2.x."""

    def __init__(self, num_units, num_features, dtype=tf.float32):
        """Initialize the model with LSTM and output layers.
        Args:
            num_units: Number of units in the LSTM layer.
            num_features: Dimensionality of time series (features per timestep).
            dtype: The floating point data type to use.
        """
        super(LSTMTimeSeriesModel, self).__init__()
        self.num_units = num_units
        self.num_features = num_features
        self.lstm_layer = layers.LSTM(num_units, return_sequences=False, dtype=dtype)
        self.dense_layer = layers.Dense(num_features, dtype=dtype)

    def call(self, inputs, states=None):
        """Forward pass of the model."""
        # LSTM takes inputs of shape (batch_size, time_steps, num_features)
        lstm_output = self.lstm_layer(inputs)
        predictions = self.dense_layer(lstm_output)
        return predictions

    def get_start_state(self):
        """Return the initial state for the LSTM cell."""
        return [
            tf.zeros([self.num_units], dtype=self.dtype),
            tf.zeros([self.num_units], dtype=self.dtype)
        ]

    def filtering_step(self, current_times, current_values, state, predictions):
        """Update the model state based on observations."""
        transformed_values = self._transform(current_values)
        # Mean squared error for the loss.
        loss = tf.reduce_mean(tf.square(predictions - transformed_values), axis=-1)
        # Update state tuple (here it just includes current values and time).
        new_state_tuple = (current_times, transformed_values)
        return new_state_tuple, {"loss": loss}

    def prediction_step(self, current_times, state):
        """Advance the LSTM state using the previous observation or prediction."""
        previous_observation, lstm_state = state
        lstm_output = self.lstm_layer(previous_observation, initial_state=lstm_state)
        next_prediction = self.dense_layer(lstm_output)
        return next_prediction

    def _transform(self, data):
        """Normalize the data."""
        mean, variance = self._input_statistics
        return (data - mean) / variance

    def _de_transform(self, data):
        """De-normalize the data."""
        mean, variance = self._input_statistics
        return data * variance + mean

    def train_model(self, data, epochs=100):
        """Compile and train the model."""
        self.compile(optimizer="adam", loss="mse")
        self.fit(data, epochs=epochs)

# Example usage:
# Create model instance
num_units = 64
num_features = 1  # For univariate time series
model = LSTMTimeSeriesModel(num_units=num_units, num_features=num_features)

# Model training example
# train_data should be a TensorFlow dataset of time series data
# model.train_model(train_data)
