import os
# Suppress TensorFlow informational and warning messages, only errors will be printed.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sionna.mapping import Mapper, Demapper
from sionna.utils import hard_decisions

# Function that converts a complex tensor into a real-valued tensor
def complex_to_real(x_complex):
    return tf.concat([tf.math.real(x_complex), tf.math.imag(x_complex)], axis=1)

# Function that converts channel matrices into their equivalent real-valued representation
def H_complex_to_real(H_complex):
    H_real = tf.math.real(H_complex)
    H_imag = tf.math.imag(H_complex)
    top = tf.concat([H_real, -H_imag], axis=2)
    bottom = tf.concat([H_imag, H_real], axis=2)
    return tf.concat([top, bottom], axis=1) #(2*Nr_complex, 2*Nt_complex)

# DetNetModel: A Deep Unfolding MIMO Detector
@tf.keras.utils.register_keras_serializable(package="Custom", name="DetNetModel")
class DetNetModel(tf.keras.Model):
    def __init__(self, Nr, Nt, DetNet_layer, constellation):
        super(DetNetModel, self).__init__()
        self.Nr = Nr
        self.Nt = Nt
        self.L = DetNet_layer  # Number of unfolded layers (iterations)
        # Store the constellation as a constant tensor (used later for demodulation)
        self.constellation = tf.constant(constellation, dtype=tf.float32)
        
        # Create a list of Dense layers to perform the per-iteration linear operations.
        self.unfold_layers = []
        for _ in range(self.L):
            dense_layer = tf.keras.layers.Dense(
                2 * self.Nt,  # Output dimension equals twice the number of transmit antennas (for real & imaginary parts)
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.keras.initializers.Constant(0.001)
            )
            self.unfold_layers.append(dense_layer)
        
        # A trainable scalar parameter 't' used in the custom nonlinearity.
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
    @classmethod
    def from_config(cls, config):
        config['constellation'] = np.array(config['constellation'], dtype=np.float32)
        return cls(**config)

    # Perform a batched matrix-vector multiplication.
    def batch_matvec_mul(self, A, b, transpose_a=False):
        # Expand dimensions of b to perform matrix multiplication.
        b_exp = tf.expand_dims(b, axis=2)
        C = tf.matmul(A, b_exp, transpose_a=transpose_a)
        # Squeeze out the last dimension to return a vector.
        return tf.squeeze(C, axis=-1)
    
    # Forward pass of the network.
    def call(self, inputs):
        # Extract H_r and y_r from inputs (ignore x_target if it is provided).
        if isinstance(inputs, (list, tuple)):
            H_r, y_r = inputs[:2]
        else:
            raise ValueError("Inputs must be tuple/list")
        
        batch_size = tf.shape(y_r)[0]
        # Compute H^T * y (equivalent to the matched filter output)
        Hty = self.batch_matvec_mul(H_r, y_r, transpose_a=True)
        # Compute H^T * H
        HtH = tf.matmul(H_r, H_r, transpose_a=True)
        # Create an identity matrix (scaled to match the dimensions) for numerical stability
        eye = tf.eye(2 * self.Nt, batch_shape=[batch_size])
        # Invert the matrix (adding a small regularization term for stability)
        HtH_inv = tf.linalg.inv(HtH + 1e-6 * eye)
        # Compute the zero-forcing (ZF) initial estimate: xhat0 = (H^T * H)^{-1} * H^T * y
        xhat0 = tf.squeeze(tf.matmul(HtH_inv, tf.expand_dims(Hty, axis=2)), axis=-1)
        
        # Iterative detection: refine the initial estimate through the unfolded layers.
        xhat = xhat0
        for i in range(self.L):
            # Compute H^T * H * xhat for the current estimate.
            HtH_xhat = self.batch_matvec_mul(HtH, xhat)
            # Concatenate the intermediate results: H^T * y, the current estimate, and H^T * H * xhat.
            concat_input = tf.concat([Hty, xhat, HtH_xhat], axis=1)
            # Pass the concatenated input through the i-th Dense layer.
            xhat = self.unfold_layers[i](concat_input)
            # Apply the custom nonlinearity:
            xhat = tf.math.tanh(self.t * xhat)

        
        # Return the final estimate and also the second element (loss) is None because we are in inference mode.
        return xhat, None

# This function loads the pre-trained model, simulates a MIMO communication system,
# performs detection using the DetNet model, and evaluates the Bit Error Rate (BER)
# over a range of SNR values.

def main():
    detnet = tf.keras.models.load_model(
        "detnet_qpsk_tanh.keras",
        custom_objects={'DetNetModel': DetNetModel},
        compile=False
    )

    # System Parameters (must match training parameters)
    Nt_complex = 4     # Number of complex transmit antennas
    Nr_complex = 8     # Number of complex receive antennas
    num_bits = 2       # Bits per modulation symbol (QPSK)
    eval_batch_size = 100000  # Number of samples per evaluation batch

    # Mapper: Converts bits into complex modulation symbols.
    mapper = Mapper("qam", num_bits)
    # Demapper: Provides approximate log-likelihood ratios (LLRs) for detection.
    demapper = Demapper("app", "qam", num_bits_per_symbol=num_bits)

    # Evaluate the Bit Error Rate (BER) for a given SNR (in dB).
    def evaluate_ber(snr_db):
        # Generate random bits for each transmit antenna.
        bits = tf.random.uniform([eval_batch_size, Nt_complex, num_bits], 
                                 maxval=2, dtype=tf.int32)
        # Map the bits to complex modulation symbols.
        x = mapper(bits)
        # Remove any extra dimensions
        x = tf.squeeze(x, -1)

        # Create a random complex channel matrix H for each sample.
        H = tf.complex(
            tf.random.normal([eval_batch_size, Nr_complex, Nt_complex], dtype=tf.float32),
            tf.random.normal([eval_batch_size, Nr_complex, Nt_complex], dtype=tf.float32)
        )

        # Compute the noise variance based on the specified SNR.
        # The noise variance is scaled by the number of transmit antennas.
        noise_var = tf.cast(10 ** (-snr_db / 10) * Nt_complex, tf.float32)
        # Standard deviation for the noise (split equally between real and imaginary parts).
        noise_std = tf.sqrt(noise_var / 2.0)
        # Generate complex Gaussian noise.
        noise = tf.complex(
            tf.random.normal([eval_batch_size, Nr_complex]) * noise_std,
            tf.random.normal([eval_batch_size, Nr_complex]) * noise_std
        )
        
        # Compute the noiseless received signal: y = H * x.
        # Expand x to have a trailing singleton dimension for matrix multiplication.
        y = tf.squeeze(tf.matmul(H, tf.expand_dims(x, -1)), axis=-1)
        # Add noise to simulate the realistic channel.
        y = y + noise
        
        # Convert the complex channel matrix and received signal to their real representations.
        H_real = H_complex_to_real(H)
        y_real = complex_to_real(y)
        # Pass the real-valued channel and received signal to the DetNet model.
        x_hat, _ = detnet([H_real, y_real])
        # Convert the network output (which is in real domain) back to complex domain.
        # The first half corresponds to the real part, and the second half corresponds to the imaginary part.
        x_hat_complex = tf.complex(x_hat[:, :Nt_complex], x_hat[:, Nt_complex:])
        
        # Demapper: Compute log-likelihood ratios (LLRs) for the estimated symbols.
        llr = demapper([x_hat_complex, noise_var])
        # Make hard decisions (0 or 1) based on the LLR values.
        bits_hat = hard_decisions(llr)
        
        # Convert the estimated bits to int32 and flatten them to compare with the transmitted bits.
        bits_hat = tf.cast(bits_hat, tf.int32)
        bits_flat = tf.reshape(bits, [eval_batch_size, -1])
        bits_hat_flat = tf.reshape(bits_hat, [eval_batch_size, -1])
        
        # Compute the BER: the fraction of bits that do not match.
        ber = tf.reduce_mean(tf.cast(tf.not_equal(bits_flat, bits_hat_flat), tf.float32))
        # Return the BER as a NumPy scalar.
        return ber.numpy()

    # Create a range of SNR values (in dB) to test.
    snr_values = np.arange(0, 10.5, 0.5)
    ber_vals = []  # List to store the BER for each SNR

    # Loop over each SNR value, evaluate the BER, and print the result.
    for snr in snr_values:
        ber_val = evaluate_ber(snr)
        print(f"SNR = {snr:.1f} dB, BER = {ber_val:.5f}")
        ber_vals.append(ber_val)
    
    # Plot the BER vs. SNR curve on a semilogarithmic scale.
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_values, ber_vals, 'o-')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("4x8 MIMO QPSK Detection Performance")
    plt.grid(True)
    plt.show()
    
if __name__ == "__main__":
    main()
