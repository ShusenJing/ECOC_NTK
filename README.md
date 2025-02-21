Error Correction Output Codes for Robust Neural Networks against Weight-errors: A Neural Tangent Kernel Point of View

Recent works adopt error correction output codes (ECOCs) as a replacement of one-hot code to combat the errors happened in deep neural network (DNN) accelerator due to hardware defects. Despite of numerical success and intuitions, the fundamentals efficacy of ECOCs on the DNNs are still not well understood.  To fill the gap, in this paper, we make theoretical analysis on ECOCs in the regime of neural tangent kernels (NTKs). We prove that, when the devices are error-free, using ECOCs (rather than one-hot code) is equivalent to changing decoding metric to a corresponding vector $A$-norm (from $l_2$ norm). We also prove that, if the scale of weight-errors, resulting from hardware defects, is smaller than a threshold determined by the ratio of Hamming distance to code length, then the NN can achieve  almost perturbation-free performance when code length is large enough. Based on the theoretical findings, we proposed two ECOC construction methods targeting on small and large scale classification tasks, respectively. For small scale tasks, we use a NN to parameterize the ECOC and directly optimize the trade-off between orthogonality and Hamming distance during code construction. For larger scale tasks, we search for the trade-off by picking up each codeword from either Hadamard codes or their complementary. We made extensive experiments demonstrating the superior performance of proposed codes.

paper link: https://openreview.net/forum?id=7LIm53Jiic&referrer=%5Bthe%20profile%20of%20Wujie%20Wen%5D(%2Fprofile%3Fid%3D~Wujie_Wen2)

Environment requirements:
Python 3.11.9
PyTorch 2.3.0

ECOCs are trained and evaluated in the main files. 
find_code_dnn.py is used to construct ECOC using Method 1.
ECOC constructed by Method 2 can be obtained from Hardmard_code_list function in code_construct.py
