## Error Correction Output Codes for Robust Neural Networks against Weight-errors: A Neural Tangent Kernel Point of View

Recent works adopt error correction output codes (ECOCs) as a replacement of one-hot code to combat the errors happened in deep neural network (DNN) accelerator due to hardware defects. Despite of numerical success and intuitions, the fundamentals efficacy of ECOCs on the DNNs are still not well understood.  To fill the gap, in this paper, we make theoretical analysis on ECOCs in the regime of neural tangent kernels (NTKs). We prove that, when the devices are error-free, using ECOCs (rather than one-hot code) is equivalent to changing decoding metric to a corresponding vector $A$-norm (from $l_2$ norm). We also prove that, if the scale of weight-errors, resulting from hardware defects, is smaller than a threshold determined by the ratio of Hamming distance to code length, then the NN can achieve  almost perturbation-free performance when code length is large enough. Based on the theoretical findings, we proposed two ECOC construction methods targeting on small and large scale classification tasks, respectively. For small scale tasks, we use a NN to parameterize the ECOC and directly optimize the trade-off between orthogonality and Hamming distance during code construction. For larger scale tasks, we search for the trade-off by picking up each codeword from either Hadamard codes or their complementary. We made extensive experiments demonstrating the superior performance of proposed codes.

paper link: https://openreview.net/forum?id=7LIm53Jiic&referrer=%5Bthe%20profile%20of%20Wujie%20Wen%5D(%2Fprofile%3Fid%3D~Wujie_Wen2)

Environment requirements:
Python 3.11.9
PyTorch 2.3.0



ECOCs are trained and evaluated in the main files. 
find_code_dnn.py is used to construct ECOC using Method 1.
ECOC constructed by Method 2 can be obtained from Hardmard_code_list function in code_construct.py


### Method1: Direct Optimization

Let $Z \in \{-1, 1\}^{n_L \times C}$ be the ECOC codebook, i.e., a horizontal stack of codewords. Then we construct the code by solving the following optimization problem:

$$
\min_{Z \in \{-1, 1\}^{n_L \times C}} -\sum_{i \neq j} \|Z[i] - Z[j]\|^2 + \lambda \left( \sum_{i \neq j}(Z[i]^T Z[j])^2 - \beta \sum_i \|Z[i]\|^2 \right)
$$

- The first term: **pair-wise codeword distance**.
- The second term: **correlation**, which penalizes the magnitude of off-diagonal elements while promoting the amplitude of diagonal elements in the correlation matrix $Z^T Z$.

Before employing standard optimization algorithms, it is necessary to relax the feasible set from the discrete binary domain $\{-1, 1\}^{n_L \times C}$ to the continuous interval $[-1, 1]^{n_L \times C}$. To further eliminate these box constraints, we reparameterize $Z$ with $\tanh(Z')$. This allows the application of gradient descent to effectively solve the optimization problem. Note that the problem is generally convex, so we run the gradient descent multiple times with random initialization and choose the solution with the smallest loss.

### Method2: Picking from Hadamard

In Method 2, we pick up $C$ codewords from Hadamard codes and their complementary codes. Without loss of generality, decompose $C$ into $C = 2a + b$, where $a$ and $b$ are non-negative integers. Let $\{v_1, v_2, \dots, v_H\}$ be the Hadamard codewords, then the codewords are $\{v_1, v_2, \dots, v_a, -v_1, -v_2, \dots, -v_a, v_{a+1}, v_{a+2}, \dots, v_{a+b}\}$.

We can observe that for a codeword $\mathcal{E}^{pick}_{a,b}(i) = v_i$ with $i \leq a$, it is orthogonal to all other codewords except $\mathcal{E}^{pick}_{a,b}(i + a) = -v_i$. In addition, the Hamming distance between it and other codewords is half of the code length, except for $\mathcal{E}^{pick}_{a,b}(i + a)$, where the Hamming distance is the code length.


