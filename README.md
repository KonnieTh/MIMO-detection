# MIMO-detection

# Results

The following tables show example BER curves for:

QPSK vs. 16QAM modulation
tanh vs. custom ReLU activation functions

QPSK (tanh) Performance
SNR (dB)	BER
0.0	     0.05485
0.5	    0.04602
1.0	    0.03816
1.5	    0.03166
2.0	    0.02570
2.5	    0.02036
3.0	    0.01624
3.5	    0.01277
4.0	    0.00965
4.5	    0.00723
5.0	    0.00566
5.5	    0.00393
6.0	    0.00290
6.5	    0.00200
7.0	    0.00152
7.5	    0.00108
8.0	    0.00066
8.5	    0.00048
9.0	    0.00030
9.5	    0.00020
10.0	  0.00015

QPSK (Custom ReLU) Performance
SNR (dB)	BER
0.0	    0.05253
0.5	    0.04358
1.0	    0.03653
1.5	    0.02932
2.0	    0.02335
2.5	    0.01865
3.0	    0.01422
3.5	    0.01088
4.0	    0.00842
4.5	    0.00606
5.0	    0.00439
5.5	    0.00316
6.0	    0.00239
6.5	    0.00149
7.0	    0.00107
7.5	    0.00070
8.0	    0.00051
8.5	    0.00037
9.0	    0.00021
9.5	    0.00014
10.0	  0.00011

16QAM (Custom ReLU) Performance
SNR (dB)	BER
0.0	    0.31234
0.5	    0.30836
1.0	    0.30446
1.5	    0.30070
2.0	    0.29666
2.5	    0.29321
3.0	    0.29039
3.5	    0.28562
4.0	    0.16884
4.5	    0.14867
5.0	    0.12941
5.5	    0.11062
6.0	    0.09271
6.5	    0.07671
7.0	    0.06278
7.5	    0.05003
8.0	    0.04000
8.5	    0.03188
9.0	    0.02498
9.5	    0.01959
10.0	  0.01568

16QAM (tanh) Performance
SNR (dB)	BER
0.0	    0.31368
0.5	    0.30984
1.0	    0.30552
1.5	    0.30189
2.0	    0.29813
2.5	    0.29517
3.0	    0.29122
3.5	    0.28817
4.0	    0.28431
4.5	    0.24837
5.0	    0.20208
5.5	    0.16037
6.0	    0.12607
6.5	    0.09889
7.0	    0.07862
7.5	    0.06254
8.0	    0.05049
8.5	    0.04153
9.0	    0.03471
9.5	    0.02997
10.0	  0.02618

# Comments on Results

QPSK vs. 16QAM

QPSK achieves lower BER at a given SNR compared to 16QAM, as expected with smaller constellations.
16QAM consistently requires a higher SNR to reach the same BER floor as QPSK.

Impact of Activation Function (tanh vs. ReLU)

For QPSK, the performance of tanh vs. the custom ReLU is quite close, though the custom ReLU is marginally better at higher SNR.
For 16QAM, the custom ReLU generally outperforms tanh for moderate and high SNR values. 
