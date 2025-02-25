import os
# Suppress TensorFlow informational and warning messages, only errors will be printed.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sionna.mapping import Mapper, Demapper
from sionna.utils import hard_decisions
from sionna.mimo import lmmse_equalizer


# Function that converts a complex tensor into a real-valued tensor
def complex_to_real(x_complex):
    return tf.concat([tf.math.real(x_complex), tf.math.imag(x_complex)], axis=1)

# Function that converts complex channel matrices into their equivalent real-valued representation
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
        self.L = DetNet_layer
        # Store the constellation as a constant tensor.
        self.constellation = tf.constant(constellation, dtype=tf.float32)
        
        # Create a list of Dense layers (one per iteration) for unfolding.
        self.unfold_layers = []
        for _ in range(self.L):
            dense_layer = tf.keras.layers.Dense(
                2 * self.Nt,  # Output dimension is 2*Nt to account for real and imaginary parts.
                activation=None,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.keras.initializers.Constant(0.001)
            )
            self.unfold_layers.append(dense_layer)
        # A trainable scalar parameter used in the custom nonlinearity.
        self.t = tf.Variable(0.5, trainable=True, dtype=tf.float32)

    # Return a configuration dictionary for model serialization.
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Nr': self.Nr,
            'Nt': self.Nt,
            'DetNet_layer': self.L,
            # Constellation is converted to list for JSON serialization.
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
        b_exp = tf.expand_dims(b, axis=2)  # Expand b to shape [batch_size, N, 1].
        C = tf.matmul(A, b_exp, transpose_a=transpose_a)
        return tf.squeeze(C, axis=-1)  # Remove the singleton dimension.
    
    # Forward pass of the network.
    def call(self, inputs, training=False):
        # Extract H_r and y_r; ignore any extra inputs.
        if isinstance(inputs, (list, tuple)):
            H_r, y_r = inputs[:2]
        else:
            raise ValueError("Inputs must be tuple/list")
        
        batch_size = tf.shape(y_r)[0]
        # Compute H^T * y.
        Hty = self.batch_matvec_mul(H_r, y_r, transpose_a=True)
        # Compute H^T * H.
        HtH = tf.matmul(H_r, H_r, transpose_a=True)
        # Create an identity matrix for regularization.
        eye = tf.eye(2 * self.Nt, batch_shape=[batch_size])
        # Compute the inverse of H^T * H (with a small regularization term for stability).
        HtH_inv = tf.linalg.inv(HtH + 1e-6 * eye)
        # Zero-forcing initial estimate: xhat0 = (H^T*H)^(-1) * (H^T*y)
        xhat0 = tf.squeeze(tf.matmul(HtH_inv, tf.expand_dims(Hty, axis=2)), axis=-1)
        
        # Iteratively refine the estimate using the unfolded layers.
        xhat = xhat0
        for i in range(self.L):
            # Compute H^T * H * xhat.
            HtH_xhat = self.batch_matvec_mul(HtH, xhat)
            # Concatenate [Hty, xhat, HtH_xhat] as input to the current layer.
            concat_input = tf.concat([Hty, xhat, HtH_xhat], axis=1)
            # Pass the concatenated input through the current Dense layer.
            xhat = self.unfold_layers[i](concat_input)
            # Apply the custom nonlinearity:
            # f(x) = -1 + ReLU(x+t)/|t| - ReLU(x-t)/|t|
            t_abs = tf.abs(self.t) + 1e-8  # Add epsilon to avoid division by zero.
            xhat = -1.0 + tf.nn.relu(xhat + self.t)/t_abs - tf.nn.relu(xhat - self.t)/t_abs
        
        # Return the final estimate and also the second element (loss) is None because we are in inference mode.
        return xhat, None

# Configuration and System Parameters
Nt_complex = 4       # Number of complex transmit antennas.
Nr_complex = 8       # Number of complex receive antennas.
num_bits = 2         # Bits per symbol (QPSK uses 2 bits).
eval_batch_size = 100000  # Number of samples per SNR point.

# Instantiate the Mapper and Demapper for QPSK.
mapper = Mapper("qam", num_bits)
demapper = Demapper("app", "qam", num_bits_per_symbol=num_bits)

# Load the pre-trained DetNet model from file.
detnet = tf.keras.models.load_model(
    "detnet_qpsk_relu.keras",
    custom_objects={'DetNetModel': DetNetModel},
    compile=False
)

# Simulate a batch of data for a given SNR (in dB) and compute the BER using DetNet.
def simulate_ber_detnet_for_snr(snr_db):
    # 1) Random bits generation and mapping to QPSK symbols.
    bits = tf.random.uniform([eval_batch_size, Nt_complex, num_bits],
                               maxval=2, dtype=tf.int32)
    x_complex = mapper(bits)
    # remove any extra dimensions.
    x_complex = tf.squeeze(x_complex, -1)
    
    # 2) Generate a random complex channel matrix.
    H = tf.complex(
        tf.random.normal([eval_batch_size, Nr_complex, Nt_complex]),
        tf.random.normal([eval_batch_size, Nr_complex, Nt_complex])
    )

    # 3) Generate AWGN noise.
    # Compute noise variance (scaled by the number of transmit antennas).
    noise_var = tf.cast(10.0 ** (-snr_db/10.0) * Nt_complex, tf.float32)
    noise_std = tf.sqrt(noise_var / 2.0)
    noise = tf.complex(
        tf.random.normal([eval_batch_size, Nr_complex]) * noise_std,
        tf.random.normal([eval_batch_size, Nr_complex]) * noise_std
    )

    # 4) Compute the received signal.
    y_complex = tf.squeeze(tf.matmul(H, tf.expand_dims(x_complex, -1)), axis=-1)
    y_complex += noise

    # 5) Use DetNet for detection.
    # Convert complex channel and received signal to real-valued representations.
    H_r = H_complex_to_real(H)
    y_r = complex_to_real(y_complex)
    # Get the output from the DetNet model.
    xhat_real, _ = detnet([H_r, y_r])
    # Convert the real-domain output back to complex numbers.
    xhat_complex = tf.complex(xhat_real[:, :Nt_complex], xhat_real[:, Nt_complex:])

    # 6) Demap the detected symbols to LLRs and then to bits.
    llr = demapper([xhat_complex, noise_var])
    bits_hat = hard_decisions(llr)

    # Flatten true bits and detected bits to compare element-wise.
    bits_flat = tf.reshape(bits, [eval_batch_size, -1])
    bits_hat_flat = tf.reshape(tf.cast(bits_hat, tf.int32), [eval_batch_size, -1])
    # Compute BER as the average fraction of mismatched bits.
    ber = tf.reduce_mean(tf.cast(tf.not_equal(bits_flat, bits_hat_flat), tf.float32))
    return ber.numpy()

# Simulate a batch of data for a given SNR (in dB) and compute the BER using LMMSE detection.
def simulate_ber_lmmse_for_snr(snr_db):
    # 1) Generate random bits and map them to QPSK symbols.
    bits = tf.random.uniform([eval_batch_size, Nt_complex, num_bits],
                               maxval=2, dtype=tf.int32)
    x_complex = mapper(bits)
    x_complex = tf.squeeze(x_complex, -1) / tf.complex(tf.sqrt(2.0), 0.0)
    
    # 2) Generate a random complex channel.
    H = tf.complex(
        tf.random.normal([eval_batch_size, Nr_complex, Nt_complex]),
        tf.random.normal([eval_batch_size, Nr_complex, Nt_complex])
    )

    # 3) Generate AWGN noise.
    noise_var = tf.cast(10.0 ** (-snr_db/10.0) * Nt_complex, tf.float32)
    noise_std = tf.sqrt(noise_var / 2.0)
    noise = tf.complex(
        tf.random.normal([eval_batch_size, Nr_complex]) * noise_std,
        tf.random.normal([eval_batch_size, Nr_complex]) * noise_std
    )

    # 4) Compute the received signal y = H * x + n.
    y_complex = tf.squeeze(tf.matmul(H, tf.expand_dims(x_complex, -1)), axis=-1)
    y_complex += noise

    # 5) LMMSE detection:
    # Add a small epsilon on the diagonal for numerical stability.
    epsilon = 1e-6
    I = tf.eye(Nr_complex, batch_shape=[eval_batch_size], dtype=tf.complex64)
    err_var_reg = tf.complex(epsilon, 0.0)*I
    # Call the LMMSE equalizer.
    x_hat, no_eff = lmmse_equalizer(y_complex, H, err_var_reg, noise_var)

    # 6) Demap the equalized symbols to bits.
    llr = demapper([x_hat, no_eff])
    bits_hat = hard_decisions(llr)

    bits_flat = tf.reshape(bits, [eval_batch_size, -1])
    bits_hat_flat = tf.reshape(tf.cast(bits_hat, tf.int32), [eval_batch_size, -1])
    ber = tf.reduce_mean(tf.cast(tf.not_equal(bits_flat, bits_hat_flat), tf.float32))
    return ber.numpy()

# Runs simulations over a range of SNR values for both DetNet and LMMSE detectors,
# prints the BER for each SNR, and plots the BER vs. SNR curves for comparison.
def main():
    # Define the SNR range: from 0 to 10 dB in steps of 0.5 dB.
    snr_values = np.arange(0, 10.5, 0.5)
    ber_detnet = []  # To store BER results for DetNet.
    ber_lmmse = []   # To store BER results for LMMSE.

    # Loop over each SNR value.
    for snr_db in snr_values:
        # Simulate BER using the DetNet detector.
        ber_d = simulate_ber_detnet_for_snr(snr_db)
        ber_detnet.append(ber_d)

        # Simulate BER using LMMSE detection.
        ber_l = simulate_ber_lmmse_for_snr(snr_db)
        ber_lmmse.append(ber_l)

        print(f"SNR = {snr_db:.1f} dB: DetNet BER = {ber_d:.5f}, LMMSE BER = {ber_l:.5f}")

    # Plot the BER vs. SNR curves on a semilogarithmic scale.
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_values, ber_detnet, 'o-', label='DetNet')
    plt.semilogy(snr_values, ber_lmmse, 's-', label='LMMSE')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate (BER)")
    plt.title("4x8 MIMO QPSK: DetNet vs. LMMSE")
    plt.grid(True, which="both")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
