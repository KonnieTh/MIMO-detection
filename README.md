# Deep MIMO Detection (DetNet)

This repository implements a deep MIMO detector using a neural network architecture.

**Activations**:

- **tanh** (standard activation):

  `tanh(t * x)`

  where `t` is trainable.

- **custom ReLU**, defined as:

  `f(x) = -1 + ReLU(x + t)/|t| - ReLU(x - t)/|t|`

  where `t` is trainable.



The following results show the Bit Error Rate (BER) performance across a range of SNR values for QPSK and 16QAM modulation schemes.

---

## Table of Contents

1. [QPSK Performance Comparison](#qpsk-performance-comparison)
2. [16QAM Performance Comparison](#16qam-performance-comparison)
3. [Comments on the Results](#comments-on-the-results)

---

## QPSK Performance Comparison

| SNR (dB) | tanh BER  | custom ReLU BER | custom ReLU (Quantized) BER |
|----------|----------:|----------------:|----------------------------:|
| 0.0      | 0.05485   | 0.05253         | 0.05029                     |
| 0.5      | 0.04602   | 0.04358         | 0.04205                     |
| 1.0      | 0.03816   | 0.03653         | 0.03442                     |
| 1.5      | 0.03166   | 0.02932         | 0.02763                     |
| 2.0      | 0.02570   | 0.02335         | 0.02176                     |
| 2.5      | 0.02036   | 0.01865         | 0.01735                     |
| 3.0      | 0.01624   | 0.01422         | 0.01316                     |
| 3.5      | 0.01277   | 0.01088         | 0.00980                     |
| 4.0      | 0.00965   | 0.00842         | 0.00734                     |
| 4.5      | 0.00723   | 0.00606         | 0.00532                     |
| 5.0      | 0.00566   | 0.00439         | 0.00394                     |
| 5.5      | 0.00393   | 0.00316         | 0.00272                     |
| 6.0      | 0.00290   | 0.00239         | 0.00183                     |
| 6.5      | 0.00200   | 0.00149         | 0.00138                     |
| 7.0      | 0.00152   | 0.00107         | 0.00091                     |
| 7.5      | 0.00108   | 0.00070         | 0.00060                     |
| 8.0      | 0.00066   | 0.00051         | 0.00042                     |
| 8.5      | 0.00048   | 0.00037         | 0.00028                     |
| 9.0      | 0.00030   | 0.00021         | 0.00020                     |
| 9.5      | 0.00020   | 0.00014         | 0.00013                     |
| 10.0     | 0.00015   | 0.00011         | 0.00010                     |

---

## 16QAM Performance Comparison

| SNR (dB) | tanh BER  | custom ReLU BER |
|----------|----------:|----------------:|
| 0.0      | 0.31368   | 0.31234         |
| 0.5      | 0.30984   | 0.30836         |
| 1.0      | 0.30552   | 0.30446         |
| 1.5      | 0.30189   | 0.30070         |
| 2.0      | 0.29813   | 0.29666         |
| 2.5      | 0.29517   | 0.29321         |
| 3.0      | 0.29122   | 0.29039         |
| 3.5      | 0.28817   | 0.28562         |
| 4.0      | 0.28431   | 0.16884         |
| 4.5      | 0.24837   | 0.14867         |
| 5.0      | 0.20208   | 0.12941         |
| 5.5      | 0.16037   | 0.11062         |
| 6.0      | 0.12607   | 0.09271         |
| 6.5      | 0.09889   | 0.07671         |
| 7.0      | 0.07862   | 0.06278         |
| 7.5      | 0.06254   | 0.05003         |
| 8.0      | 0.05049   | 0.04000         |
| 8.5      | 0.04153   | 0.03188         |
| 9.0      | 0.03471   | 0.02498         |
| 9.5      | 0.02997   | 0.01959         |
| 10.0     | 0.02618   | 0.01568         |

---

## Comments on the Results

- **QPSK:**  
  For the QPSK modulation scheme, both activation functions yield similar BER performance across the SNR range. However, the custom ReLU consistently produces a slightly lower BER compared to tanh.

- **16QAM:**  
  The benefits of the custom ReLU become more pronounced for the more complex 16QAM modulation. At low SNR values, both activations perform similarly; however, as SNR increases (starting around 4 dB), the tanh activation saturates, leading to higher BER. In contrast, the custom ReLU shows a marked improvement—at 4 dB, tanh yields a BER of 0.28431, while custom ReLU drops it to 0.16884.
  This gap continues through the moderate-to-high SNR range, demonstrating that the custom ReLU is better suited for handling the larger constellation and finer distinctions required by 16QAM.

---

*End of README.md*
