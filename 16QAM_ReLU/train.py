import os
# Suppress TensorFlow informational and warning messages, only errors will be printed.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sionna.mapping import Mapper, Constellation



# Set random seeds
tf.random.set_seed(42)
np.random.seed(42)


# Function that converts a complex tensor into a real-valued tensor
def complex_to_real(x_complex):
    return tf.concat([tf.math.real(x_complex), tf.math.imag(x_complex)], axis=1)

# Function that converts a complex channel matrix into the equivalent real-valued representation
def H_complex_to_real(H_complex):
    H_real = tf.math.real(H_complex)
    H_imag = tf.math.imag(H_complex)
    top = tf.concat([H_real, -H_imag], axis=2)
    bottom = tf.concat([H_imag, H_real], axis=2)
    return tf.concat([top, bottom], axis=1) #(2*Nr_complex, 2*Nt_complex)

# DetNetModel: A Deep Unfolding MIMO Detector
class DetNetModel(tf.keras.Model):
    def __init__(self, Nr, Nt, DetNet_layer, constellation):
        super(DetNetModel, self).__init__()
        self.Nr = Nr  # Number of complex receive antennas (real domain: 2*Nr)
        self.Nt = Nt  # Number of complex transmit antennas (real domain: 2*Nt)
        self.L = DetNet_layer  # Total number of unfolded iterations (layers)
        # Save the constellation as a constant tensor
        self.constellation = tf.constant(constellation, dtype=tf.float32)

        self.unfold_layers = []
        for _ in range(self.L):
            dense_layer = tf.keras.layers.Dense(
                2 * self.Nt, 
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.keras.initializers.Constant(0.001)
            )
            self.unfold_layers.append(dense_layer)
        
        # Trainable scalar parameter 't' used in the custom nonlinearity.
        self.t = tf.Variable(0.5, trainable=True, dtype=tf.float32)

    # Return a configuration dictionary for model serialization.
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Nr': self.Nr,
            'Nt': self.Nt,
            'DetNet_layer': self.L,
            # Convert the constellation tensor to a list for JSON serialization.
            'constellation': self.constellation.numpy().tolist(),
        })
        return config
    
    # Create a DetNetModel instance from a configuration dictionary.
    def from_config(cls, config):
        # Convert the constellation list back into a NumPy array.
        config['constellation'] = np.array(config['constellation'], dtype=np.float32)
        return cls(**config)
    
    # Perform a batched matrix-vector multiplication.
    def batch_matvec_mul(self, A, b, transpose_a=False):
        b_exp = tf.expand_dims(b, axis=2)  # Expand dims to shape (batch_size, N, 1)
        C = tf.matmul(A, b_exp, transpose_a=transpose_a) #transpose_a: If True, A is transposed along its last two axes.
        return tf.squeeze(C, axis=-1)  # Remove the last singleton dimension
    
    # Forward pass of the network.
    def call(self, inputs, training=False):

        # Parse inputs
        if isinstance(inputs, (list, tuple)):
            if len(inputs) == 3:
                H_r, y_r, x_target = inputs
            elif len(inputs) == 2:
                H_r, y_r = inputs
                x_target = None
            else:
                raise ValueError("Expected inputs: (H_r, y_r) or (H_r, y_r, x_target)")
        else:
            raise ValueError("Inputs must be a tuple or list")
        
        batch_size = tf.shape(y_r)[0]
        # Compute H^T * y
        Hty = self.batch_matvec_mul(H_r, y_r, transpose_a=True)
        # Compute H^T * H 
        HtH = tf.matmul(H_r, H_r, transpose_a=True)
        # Add a small regularization term (1e-6) times the identity matrix for numerical stability
        eye = tf.eye(2 * self.Nt, batch_shape=[batch_size])
        HtH_inv = tf.linalg.inv(HtH + 1e-6 * eye)
        # Compute the zero-forcing initial estimate: xhat0 = (H^T*H)^{-1} * H^T * y
        xhat0 = tf.squeeze(tf.matmul(HtH_inv, tf.expand_dims(Hty, axis=2)), axis=-1)
        
        # Initialize the iterative detection with the ZF estimate.
        xhat = xhat0
        xhat_list = []  # To store outputs from each layer
        for i in range(self.L):
            # Compute HtH * xhat
            HtH_xhat = self.batch_matvec_mul(HtH, xhat)
            # Concatenate Hty, xhat, and HtH_xhat to form the input of the unfolded layer.
            concat_input = tf.concat([Hty, xhat, HtH_xhat], axis=1)
            # Pass the concatenated input through a dense layer.
            xhat = self.unfold_layers[i](concat_input)
            # Apply the custom nonlinearity:
            #    f(x) = -1 + ReLU(x+t)/|t| - ReLU(x-t)/|t|
            t_abs = tf.abs(self.t) + 1e-8  # Add epsilon to avoid division by zero
            xhat = -1.0 + tf.nn.relu(xhat + self.t) / t_abs - tf.nn.relu(xhat - self.t) / t_abs
            # Save the output of the current iteration (layer)
            xhat_list.append(xhat)
        
        # If in training mode and the ground truth x_target is provided, compute the loss.
        if training and x_target is not None:
            loss = 0.0
            # Sum the mean squared error (MSE) losses from each layer output.
            # Each layer's MSE is weighted by the logarithm of its index.
            for i, xhat_k in enumerate(xhat_list, start=1):
                mse = tf.reduce_mean(tf.keras.losses.mean_squared_error(x_target, xhat_k))
                loss += mse * tf.math.log(tf.cast(i, tf.float32) + 1e-8)
        else:
            loss = None
        
        # Return the final estimate and the loss (if computed)
        return xhat, loss
    
    # Demodulate the network output back into symbol indices.
    def demodulate(self, x):
        # Form complex numbers from the real-valued output:
        # First Nt entries -> real part; next Nt entries -> imaginary part.
        x_complex = tf.complex(x[:, :self.Nt], x[:, self.Nt:2*self.Nt])
        # Reshape to a column vector for distance computation
        x_complex = tf.reshape(x_complex, [-1, 1])
        # Convert stored constellation points to complex numbers.
        constellation_complex = tf.complex(self.constellation[:, 0], self.constellation[:, 1])
        # Reshape so that each row of x_complex can be compared against all constellation points.
        constellation_complex = tf.reshape(constellation_complex, [1, -1])
        # Compute the Euclidean distance (absolute difference) between estimated and constellation points.
        distances = tf.abs(x_complex - constellation_complex)
        # Find the index of the closest constellation point.
        indices = tf.cast(tf.argmin(distances, axis=1), tf.int32)
        # Reshape indices to have shape (batch_size, Nt)
        return tf.reshape(indices, [-1, self.Nt])
    # Compute the accuracy (fraction of correctly demodulated symbols)
    def accuracy(self, x_true, x_pred):
        return tf.reduce_mean(tf.cast(tf.equal(x_true, x_pred), tf.float32))
    
# A utility Function for the SNR Range per Epoch
def snr_range_for_epoch(epoch):
    if epoch < 25:
        return 0.0, 5.0
    else:
        return 5.0, 10.0

# This function sets up the MIMO detection problem, creates training batches and trains the DetNetModel.
def main():
    # Define the number of complex antennas and modulation bits.
    Nt_complex = 4  # Number of complex transmit antennas
    Nr_complex = 8  # Number of complex receive antennas
    num_bits = 4    # Bits per symbol
    
    # Training parameters
    num_epochs = 50
    steps_per_epoch = 150
    train_batch_size = 2000
    
    # Create a constellation
    constellation = Constellation(
        constellation_type="qam",
        num_bits_per_symbol=num_bits,
        normalize=True,  # Preserve original amplitudes.
        center=False,     # Do not center the constellation.
        dtype=tf.complex64
    )

    # Convert the points tensor to a NumPy array
    arr = constellation.points.numpy()

    # Create a 2D NumPy array where each row is [real, imag]
    constellation_np = np.stack([np.real(arr), np.imag(arr)], axis=1)
    
    # Parameters for the DetNetModel
    params = {
        'Nr': Nr_complex,
        'Nt': Nt_complex,
        'DetNet_layer': 10,  # Number of unfolding iterations
        'constellation': constellation_np,
    }
    
    # Instantiate a mapper from Sionna. This is used to map random bits to modulation symbols.
    mapper = Mapper("qam", num_bits)
    
    # Create an instance of DetNetModel with the specified parameters.
    detnet = DetNetModel(**params)
    
    # Define the optimizer for training.
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    # Training Loop
    for epoch in range(num_epochs):
        # Get the SNR range for this epoch
        snr_min, snr_max = snr_range_for_epoch(epoch)
        for _ in range(steps_per_epoch):
            epoch_loss = 0.0  # Accumulate loss over steps
            # Generate random bits for each transmit antenna and each symbol.
            bits = tf.random.uniform(shape=[train_batch_size, Nt_complex, num_bits],
                                      minval=0, maxval=2, dtype=tf.int32)
            # Map bits to complex QAM symbols.
            x_complex = mapper(bits)
            # If the mapper adds an extra singleton dimension, remove it.
            if x_complex.shape.rank > 2:
                x_complex = tf.squeeze(x_complex, axis=-1)

            # Convert the complex transmitted signal into its real representation.
            x_target = complex_to_real(x_complex)
            
            # Generate a random complex channel matrix for each batch.
            H_complex = tf.complex(tf.random.normal([train_batch_size, Nr_complex, Nt_complex]),
                                   tf.random.normal([train_batch_size, Nr_complex, Nt_complex]))
            # Convert the complex channel matrix to the real-domain representation.
            H_r = H_complex_to_real(H_complex)

            # Sample random SNR values for each instance in the batch.
            snr_db = tf.random.uniform([train_batch_size], snr_min, snr_max)
            # Compute the noise variance, the scaling with Nt_complex is to account for the number of symbols.
            noise_var = tf.cast(10 ** (-snr_db / 10) * Nt_complex, tf.float32)
            
            # Compute the noiseless received signal: y_complex = H_complex * x_complex.
            x_exp = tf.expand_dims(x_complex, axis=-1)
            y_complex = tf.squeeze(tf.matmul(H_complex, x_exp), axis=-1)
            # Generate additive white Gaussian noise (AWGN) in both real and imaginary parts.
            noise_real = tf.sqrt(noise_var / 2.0)[:, tf.newaxis] * tf.random.normal([train_batch_size, Nr_complex])
            noise_imag = tf.sqrt(noise_var / 2.0)[:, tf.newaxis] * tf.random.normal([train_batch_size, Nr_complex])
            noise = tf.complex(noise_real, noise_imag)
            # Add noise to the received signal.
            y_complex += noise
            # Convert the received complex signal into its real representation.
            y_r = complex_to_real(y_complex)
            
            # Record gradients and perform a training step using GradientTape.
            with tf.GradientTape() as tape:
                # Forward pass
                _ , loss_val = detnet((H_r, y_r, x_target), training=True)
            # Compute gradients of the loss with respect to the model parameters.
            grads = tape.gradient(loss_val, detnet.trainable_variables)
            # Apply the gradients to update the model parameters.
            optimizer.apply_gradients(zip(grads, detnet.trainable_variables))
            epoch_loss += loss_val.numpy()  # Accumulate loss for reporting
        
        # Print the average loss for the epoch.
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {epoch_loss/steps_per_epoch:.6f}")
    
    # Save the trained model in the Keras native format.
    detnet.save("detnet_16qam_relu.keras")
    print("Model saved as detnet_16qam_relu.keras")

if __name__ == "__main__":
    main()
