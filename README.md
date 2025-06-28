# Expansive Networks
The hypothesis is that, while mathematically equivalent and contrary to modern wisdom, two consecutive matrices of sizes 100x10000 and 10000x100 with no non-linearity inbetween them actually learn more easily than one 100x100 matrix. We call the act of replacing the 100x100 matrix by the former two matrices "expansion". We propose Expansive Networks -- a family of neural networks that are trained with expansions which are then collapsed at test-time via matrix multiplication. Only one layer is expanded at a time for computational memory savings. 

## Expansive NanoGPT

```
step 0: train loss 4.2261, val loss 4.2267
iter 0: loss 4.2333, time 129783.21ms, mfu -100.00%
iter 10: loss 3.0077, time 1079.97ms, mfu 1.69%
iter 20: loss 2.7173, time 1079.40ms, mfu 1.69%
iter 30: loss 2.5994, time 1090.00ms, mfu 1.69%
iter 40: loss 2.5535, time 1090.00ms, mfu 1.69%
iter 50: loss 2.5176, time 1090.08ms, mfu 1.69%
iter 60: loss 2.5034, time 1089.94ms, mfu 1.68%
iter 70: loss 2.5018, time 1086.22ms, mfu 1.68%
iter 80: loss 2.4802, time 1080.00ms, mfu 1.68%
iter 90: loss 2.4780, time 1090.00ms, mfu 1.68%
iter 100: loss 2.4552, time 1090.00ms, mfu 1.68%
iter 110: loss 2.4571, time 1089.99ms, mfu 1.68%
iter 120: loss 2.4576, time 1080.00ms, mfu 1.68%
iter 130: loss 2.4301, time 1080.12ms, mfu 1.68%
iter 140: loss 2.4223, time 1090.10ms, mfu 1.68%
iter 150: loss 2.4094, time 1420.07ms, mfu 1.64%
iter 160: loss 2.3849, time 1280.00ms, mfu 1.62%
iter 170: loss 2.3491, time 1280.00ms, mfu 1.60%
iter 180: loss 2.2925, time 1080.01ms, mfu 1.61%
iter 190: loss 2.2544, time 1077.17ms, mfu 1.62%
iter 200: loss 2.2090, time 1080.11ms, mfu 1.63%
iter 210: loss 2.1148, time 1090.00ms, mfu 1.63%
iter 220: loss 2.0772, time 1650.18ms, mfu 1.58%
iter 230: loss 2.0351, time 1465.54ms, mfu 1.54%
iter 240: loss 1.9978, time 1289.41ms, mfu 1.53%
step 250: train loss 1.9143, val loss 2.0356
```

## Baseline NanoGPT

```
step 0: train loss 4.2874, val loss 4.2823
iter 0: loss 4.2638, time 19062.76ms, mfu -100.00%
iter 10: loss 3.1451, time 140.00ms, mfu 2.66%
iter 20: loss 2.7352, time 120.02ms, mfu 2.71%
iter 30: loss 2.6193, time 109.98ms, mfu 2.77%
iter 40: loss 2.5718, time 110.02ms, mfu 2.84%
iter 50: loss 2.5246, time 100.00ms, mfu 2.92%
iter 60: loss 2.5060, time 100.02ms, mfu 3.00%
iter 70: loss 2.4936, time 110.00ms, mfu 3.04%
iter 80: loss 2.4920, time 100.00ms, mfu 3.11%
iter 90: loss 2.4662, time 110.00ms, mfu 3.14%
iter 100: loss 2.4621, time 100.00ms, mfu 3.20%
iter 110: loss 2.4553, time 110.00ms, mfu 3.22%
iter 120: loss 2.4284, time 100.01ms, mfu 3.27%
iter 130: loss 2.4084, time 100.04ms, mfu 3.31%
iter 140: loss 2.4024, time 110.00ms, mfu 3.32%
iter 150: loss 2.3961, time 100.07ms, mfu 3.36%
iter 160: loss 2.3621, time 100.01ms, mfu 3.40%
iter 170: loss 2.3494, time 100.00ms, mfu 3.43%
iter 180: loss 2.3174, time 100.01ms, mfu 3.46%
iter 190: loss 2.2405, time 109.93ms, mfu 3.45%
iter 200: loss 2.2119, time 100.10ms, mfu 3.48%
iter 210: loss 2.1410, time 109.89ms, mfu 3.47%
iter 220: loss 2.1316, time 110.00ms, mfu 3.46%
iter 230: loss 2.0757, time 110.00ms, mfu 3.46%
iter 240: loss 2.0689, time 111.45ms, mfu 3.44%
step 250: train loss 1.9589, val loss 2.0601
```

## Expansive MNIST
```
Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.000091
Test set: Average loss: 0.0394, Accuracy: 9924/10000 (99%)
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.008987
```
## Baseline MNIST
```
Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.063018
Test set: Average loss: 0.0271, Accuracy: 9912/10000 (99%)
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.001717
```
