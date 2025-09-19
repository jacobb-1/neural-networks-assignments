
# MNIST CUDA Neural Network

This project implements a neural network for the MNIST dataset using CUDA in C++.

## Required Files
Upload the following files to the repository:

- `main.cu`
- `data.cu`
- `data.h`
- `network.cu`
- `network.h`
- `neural_net.cu`
- `neural_net.cuh`
- `utils.cu`
- `utils.h`
- `.gitignore` (optional)

## Data
The code requires the following data files:
- `mnist_train.csv`
- `mnist_test.csv`

You can download these files from [Kaggle MNIST CSV](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv) or convert the original dataset to CSV format.
Place them in the same folder as the source code.

## Compilation
Make sure you have the CUDA Toolkit and a compatible compiler (e.g., NVCC) installed.

Example compilation command:
```powershell
nvcc main.cu data.cu network.cu neural_net.cu utils.cu -o mnist_cuda.exe
```

## Execution
```powershell
./mnist_cuda.exe
```

## Notes
- If you have issues with paths or dependencies, check the includes and the location of the data files.
- You can modify the code to adjust the dataset size or network parameters.

---

For questions or improvements, open an issue in the repository.
