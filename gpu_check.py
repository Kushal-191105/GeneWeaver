from numba import cuda
from src.gpu_device import print_gpu_device_info


def check_cuda():
    print("Checking CUDA environment...")
    if cuda.is_available():
        print("CUDA is available.")
        print_gpu_device_info()
    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    check_cuda()