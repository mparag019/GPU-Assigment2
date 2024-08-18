# GPU-Assigment2

# CUDA-Based 2D Convolution Operation

## Project Overview
This project implements a 2D convolution operation using CUDA, focusing on optimizing memory access patterns through memory coalescing and shared memory utilization. The convolution operation is fundamental in signal processing and image analysis, allowing for the extraction of key features from input data.

## Features
- **Memory Coalescing**: Optimized global memory access to improve performance.
- **Shared Memory Usage**: Leveraged shared memory for efficient data access and computation within CUDA blocks.
- **2D Convolution**: Implemented the core convolution operation by sliding a 2D filter over an input matrix, generating a feature map.

## Project Structure
- **starter.cu**: Contains the CUDA kernel function and main program for executing the convolution operation.
- **tester.py**: A Python script to test the correctness and performance of the implemented convolution operation.
- **input**: Folder containing sample input matrices and filters for testing.

## How to Run
1. Place your `.cu` file in the `code` folder.
2. Compile the CUDA code:
    ```bash
    nvcc -o dkernel starter.cu
    ```
3. Execute the compiled program:
    ```bash
    ./dkernel < input.txt
    ```
4. Run the provided Python tester to validate the implementation:
    ```bash
    python tester.py
    ```

## Requirements
- **CUDA Toolkit**: Ensure you have CUDA installed on your system.
- **Python 3**: Required for running the test script.

## Performance Metrics
The project measures the execution time of the convolution operation, highlighting the performance benefits achieved through optimization techniques.

## License
This project is licensed under the MIT License.
