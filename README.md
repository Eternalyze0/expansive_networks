# Expansive Networks

The hypothesis is that, while mathematically equivalent and contrary to modern wisdom, two consecutive matrices of sizes 100x10000 and 10000x100 with no non-linearity inbetween them actually learn more easily than one 100x100 matrix. We call the act of replacing the 100x100 matrix by the former two matrices "expansion". We propose Expansive Networks -- a family of neural networks that are trained with expansions which are then collapsed at test-time via matrix multiplication. Only one layer is expanded at a time for computational memory savings. 

## Expansive MNIST

Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.000091

Test set: Average loss: 0.0394, Accuracy: 9924/10000 (99%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.008987

## Baseline MNIST

Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.063018

Test set: Average loss: 0.0271, Accuracy: 9912/10000 (99%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.001717
