from numba import cuda


def check_cuda():
    print("Checking CUDA...")

    if cuda.is_available():
        print("CUDA is available.")
        print("GPU detected successfully.")

        for gpu in cuda.gpus:
            print("GPU:", gpu)

    else:
        print("CUDA is not available.")


if __name__ == "__main__":
    check_cuda()