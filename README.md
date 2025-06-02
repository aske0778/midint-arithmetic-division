# Efficient GPU Implementation of Multi-Precision Integer Division

Efficient arithmetic on multi-precision integers is a cornerstone of many scientific
and cryptographic applications that require computations on integers that exceed
the native sizes supported by modern processors. While GPU-efficient addition and
multiplication has been well explored, division has been subject to less attention
due to its greater algorithmic complexity. This thesis attempts to bridge this gap
by implementing a GPU-efficient division, that works on integers up to 250.000 bits
in size which fit in a single cuda block, exploiting the temporal data reuse of fast
scratchpad memory. The algorithm is based on the Newton-inspired method for
computing the reciprocal of the divisor presented by Watt in [1], which performs
exact division entirely within the integer domain. Our main product is an efficient
implementation in cuda, although not outperforming the popular cgbn library,
it demonstrates promising scalability results. Moreover, to our knowledge, we are
the first to implement a parallel division capable of operating on inputs larger than
215 bits. Finally, we implement a Futhark version to explore the practical aspects
of using a high-level functional language, and conclude that current compiler
limitations introduce considerable overheads and scalability issues.

---

This repository contains the code corresponding to the paper "Efficient GPU Implementation of Multi-Precision Integer Division".
It extends existing implementations for addition and multiplication by [2] with an equivalent for division based on the shifted
inverse algorithm by [1]. Implementations have been made in CUDA and Futhark which use existing frameworks by [2] and [3] respectively.


## Usage
In order to replicate our results, simply navigate to the corresponding directory and run the corresponding Makefile:

### CUDA
The CUDA results including all divisions and gcd computation are replicated by running:

```sh
cd div/cuda
make
```

### Futhark
Similarly, all results for Futhark can be run like so:

```sh
cd div/futhark
make
```

### CGBN
The CGBN results are divided based on the size of the input integers. The results for 2^15-bit integers are run like so:

```sh
cd cuda/cgbn-tests
make
```

To run the tests on smaller integers, run ``make run-X`` supplimenting for the required number of bits when divided by 32. i.e. 

```sh
make run-512
```


## References
[1]: Stephen M. Watt, ``Efficient Generic Quotients using Exact Arithmetic``, 2023 

[2]: Cosmin E. Oancea and Stephen Watt, ``GPU Implementations for Midsize Integer Addition and Multiplication``, 2024, arXiv:2405.14642 https://arxiv.org/abs/2405.14642

[3]: Thorbjørn B. Bringgaard, ``Efficient Big Integer Arithmetic using GPGPUs``, 2024


