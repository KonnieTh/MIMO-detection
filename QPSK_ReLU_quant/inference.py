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

# Custom quantized dense layer implementation
class QuantizedDense(tf.keras.layers.Layer):
    def __init__(self, units, kernel_initializer, bias_initializer, **kwargs):
        super(QuantizedDense, self).__init__(**kwargs)
        self.units = units
        # Internal dense layer without activation
        self.dense = tf.keras.layers.Dense(
            units, 
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            activation=None
        )
        
    def build(self, input_shape):
        self.dense.build(input_shape)
        self.built = True
        
    def call(self, inputs):
        # Quantize weights using 8-bit precision
        kernel = self.dense.kernel
        quantized_kernel = tf.quantization.fake_quant_with_min_max_vars(
            kernel,
            min=tf.reduce_min(kernel),
            max=tf.reduce_max(kernel),
            num_bits=8,
            narrow_range=False
        )
        # Compute linear transformation
        output = tf.matmul(inputs, quantized_kernel) + self.dense.bias
        # Quantize activations using per-batch dynamic range
        min_a = tf.reduce_min(output)
        max_a = tf.reduce_max(output)
        return tf.quantization.fake_quant_with_min_max_vars(
            output, 
            min=min_a, 
            max=max_a,
            num_bits=8,
            narrow_range=False
        )

# DetNetModel: A Deep Unfolding MIMO Detector (Quantized version)
@tf.keras.utils.register_keras_serializable(package="Custom", name="DetNetModel")
class DetNetModel(tf.keras.Model):
    def __init__(self, Nr, Nt, DetNet_layer, constellation):
        super(DetNetModel, self).__init__()
        self.Nr = Nr  # Number of complex receive antennas (real domain: 2*Nr)
        self.Nt = Nt  # Number of complex transmit antennas (real domain: 2*Nt)
        self.L = DetNet_layer  # Number of unfolded layers (iterations)
        # Store the constellation as a constant tensor (used later for demodulation)
        self.constellation = tf.constant(constellation, dtype=tf.float32)
        
        # Create quantized dense layers for each iteration
        self.unfold_layers = []
        for _ in range(self.L):
            quantized_dense = QuantizedDense(
                2 * self.Nt,
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.keras.initializers.Constant(0.001)
            )
            self.unfold_layers.append(quantized_dense)
        
        # A trainable scalar parameter 't' used in the custom nonlinearity
        self.t = tf.Variable(0.5, trainable=True, dtype=tf.float32)
   
    # Return a configuration dictionary for model serialization
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'Nr': self.Nr,
            'Nt': self.Nt,
            'DetNet_layer': self.L,
            'constellation': self.constellation.numpy().tolist(),
        })
        return config
    
    # Create instance from configuration dictionary
    @classmethod
    def from_config(cls, config):
        config['constellation'] = np.array(config['constellation'], dtype=np.float32)
        return cls(**config)

    # Perform batched matrix-vector multiplication
    def batch_matvec_mul(self, A, b, transpose_a=False):
        b_exp = tf.expand_dims(b, axis=2)
        C = tf.matmul(A, b_exp, transpose_a=transpose_a)
        return tf.squeeze(C, axis=-1)
    
    # Forward pass of the network
    def call(self, inputs):
        # Extract H_r and y_r from inputs
        if isinstance(inputs, (list, tuple)):
            H_r, y_r = inputs[:2]
        else:
            raise ValueError("Inputs must be tuple/list")
        
        batch_size = tf.shape(y_r)[0]
        # Compute H^T * y (equivalent to matched filter output)
        Hty = self.batch_matvec_mul(H_r, y_r, transpose_a=True)
        # Compute H^T * H
        HtH = tf.matmul(H_r, H_r, transpose_a=True)
        # Create identity matrix for numerical stability
        eye = tf.eye(2 * self.Nt, batch_shape=[batch_size])
        # Invert regularized matrix
        HtH_inv = tf.linalg.inv(HtH + 1e-6 * eye)
        # Compute zero-forcing initial estimate
        xhat0 = tf.squeeze(tf.matmul(HtH_inv, tf.expand_dims(Hty, axis=2)), axis=-1)
        
        # Iterative detection with quantization
        xhat = xhat0
        for i in range(self.L):
            # Compute H^T * H * xhat for current estimate
            HtH_xhat = self.batch_matvec_mul(HtH, xhat)
            # Concatenate intermediate results
            concat_input = tf.concat([Hty, xhat, HtH_xhat], axis=1)
            # Process through quantized dense layer
            xhat = self.unfold_layers[i](concat_input)
            # Apply custom nonlinearity with output quantization
            t_abs = tf.abs(self.t) + 1e-8
            xhat = -1.0 + tf.nn.relu(xhat + self.t)/t_abs - tf.nn.relu(xhat - self.t)/t_abs
            # Quantize to 8-bit fixed-point [-1, 1] range
            xhat = tf.quantization.fake_quant_with_min_max_vars(
                xhat,
                min=-1.0,
                max=1.0,
                num_bits=8,
                narrow_range=False
            )
        
        return xhat, None  # Loss is None in inference mode

# Main evaluation function
def main():
    # Load pre-trained quantized model
    detnet = tf.keras.models.load_model(
        "detnet_qpsk_relu_quantized.keras",
        custom_objects={
            'DetNetModel': DetNetModel,
            'QuantizedDense': QuantizedDense
        },
        compile=False
    )

    # System parameters (must match training configuration)
    Nt_complex = 4     # Number of complex transmit antennas
    Nr_complex = 8     # Number of complex receive antennas
    num_bits = 2       # Bits per modulation symbol (QPSK)
    eval_batch_size = 100000  # Samples per evaluation batch

    # Create mapper/demapper components
    mapper = Mapper("qam", num_bits)
    demapper = Demapper("app", "qam", num_bits_per_symbol=num_bits)

    # BER evaluation function
    def evaluate_ber(snr_db):
        # Generate random bits and map to symbols
        bits = tf.random.uniform([eval_batch_size, Nt_complex, num_bits], 
                               maxval=2, dtype=tf.int32)
        x = tf.squeeze(mapper(bits), -1)

        # Create random channel matrix
        H = tf.complex(
            tf.random.normal([eval_batch_size, Nr_complex, Nt_complex]),
            tf.random.normal([eval_batch_size, Nr_complex, Nt_complex])
        )

        # Calculate noise parameters
        noise_var = tf.cast(10 ** (-snr_db / 10) * Nt_complex, tf.float32)
        noise_std = tf.sqrt(noise_var / 2.0)
        noise = tf.complex(
            tf.random.normal([eval_batch_size, Nr_complex]) * noise_std,
            tf.random.normal([eval_batch_size, Nr_complex]) * noise_std
        )
        
        # Simulate received signal
        y = tf.squeeze(tf.matmul(H, tf.expand_dims(x, -1)), axis=-1) + noise
        
        # Convert to real-valued representation
        H_real = H_complex_to_real(H)
        y_real = complex_to_real(y)
        
        # Perform detection with quantized model
        x_hat, _ = detnet([H_real, y_real])
        
        # Demodulate to bits
        x_hat_complex = tf.complex(x_hat[:, :Nt_complex], x_hat[:, Nt_complex:])
        llr = demapper([x_hat_complex, noise_var])
        bits_hat = hard_decisions(llr)
        
        # Calculate BER
        bits_flat = tf.reshape(bits, [eval_batch_size, -1])
        bits_hat_flat = tf.reshape(tf.cast(bits_hat, tf.int32), [eval_batch_size, -1])
        return tf.reduce_mean(tf.cast(tf.not_equal(bits_flat, bits_hat_flat), tf.float32)).numpy()

    # SNR range evaluation
    snr_values = np.arange(0, 10.5, 0.5)
    ber_vals = []
    for snr in snr_values:
        ber_val = evaluate_ber(snr)
        print(f"SNR = {snr:.1f} dB, BER = {ber_val:.5f}")
        ber_vals.append(ber_val)
    
    # Plot results
    plt.figure(figsize=(8, 6))
    plt.semilogy(snr_values, ber_vals, 'o-')
    plt.xlabel("SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("4x8 MIMO QPSK Detection Performance (Quantized DetNet)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()