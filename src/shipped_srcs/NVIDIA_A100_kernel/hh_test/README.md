## About

Householder transformation mini-app for OLCF GPU Hackathon (October 2019).

## Install

* CMake (3.8+)
* Fortran compiler
* CUDA

```
mkdir build
cd build
cmake ..
```

Explicitly define Fortran compiler and CUDA installation path if necessary.

## Run

```
./hh_test b n
```

See text in [this preprint](https://arxiv.org/abs/2002.10991) for the meanings
of `b` and `n`. The value of `b` must be 2^k, where k = 1, 2, ..., 10. `n` is
typically a few thousand.
