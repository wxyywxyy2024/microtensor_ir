# src/verify.py
import os
import subprocess
import ctypes
import torch
import numpy as np

def compile_dll():
    print("--- [Step 4.1] Compiling C++ Kernel to Native Windows DLL ---")

    # Force ensure that the 'src' directory physically exists
    os.makedirs("src", exist_ok=True)

    # Check for standard Mingw-w64 g++ compiler command
    cpp_source = os.path.abspath("src/kernel.cpp")
    dll_output = os.path.abspath("src/microtensor_kernel.dll")

    # Absolute path for g++ exe. Use double backslashes '\\' so Windows paths don't break string parsing!
    compiler_path = "C:\\msys64\\ucrt64\\bin\\g++.exe"

    # WORKAROUND: If manual compilation already built it, just use it!
    if os.path.exists(dll_output):
        print(f" Found mannually compiled binary at: {dll_output}")
        print(" Bypassing background subprocess compilation layer.")
        return dll_output

    # Compilation flags optimized for maximum native speed (-O3, -mavx2)
    compile_cmd = f'g++ -O3 -shared -mavx2 "{cpp_source}" -o "{dll_output}"'

    print(f"Executing: {compile_cmd}")

    # Run using a single string command along with shell=True
    result = subprocess.run(compile_cmd, capture_output=True, text=True, shell=True)

    if result.returncode != 0:
        error_details = []
        if result.stderr: error_details.append(f"Stderr: {result.stderr.strip()}")
        if result.stdout: error_details.append(f"Stdout: {result.stdout.strip()}")
        if not error_details: error_details.append(f"OS Process exited with silent failure code: {result.returncode}")

        raise RuntimeError("Compilation Failed!\n" + "\n".join(error_details))

    print(f"Successfully generated high-performance binary: {dll_output}")
    return dll_output

def verify_correctness(dll_path):
    print("\n--- [Step 4.2] Binding DLL and Verifying Mamthematical Correctness ---")

    # 1. Load the shared library using ctypes
    kernel_lib = ctypes.CDLL(os.path.abspath(dll_path))

    # 2. Setup input tensor shapes to match our Graph IR dimensions
    # t1: 128x512, t2: 512x256, t3: 128x256
    t1_pt = torch.randn(128, 512, dtype=torch.float32)
    t2_pt = torch.randn(512, 256, dtype=torch.float32)
    t3_pt = torch.randn(128, 256, dtype=torch.float32)

    # 3. Compute Golden Reference standard via PyTorch
    # MatMul -> Add -> ReLU
    golden_output = torch.relu(torch.matmul(t1_pt, t2_pt) + t3_pt)

    # 4. Prepare empty numpy destination buffer for our custom C++ kernel
    cpp_output_np = np.zeros((128, 256), dtype=np.float32)

    # 5. Extract raw data memory pointers to pass through the ctypes FFI barrier
    t1_ptr = t1_pt.contiguous().data_ptr()
    t2_ptr = t2_pt.contiguous().data_ptr()
    t3_ptr = t3_pt.contiguous().data_ptr()

    out_ptr = cpp_output_np.ctypes.data

    # 6. Call our fused microtensor kernel natively on the hardware
    # signature: void microtensor_kernel(const float* t1, const float* t2, const float* t3, float* output)
    kernel_lib.microtensor_kernel(
        ctypes.c_void_p(t1_ptr),
        ctypes.c_void_p(t2_ptr),
        ctypes.c_void_p(t3_ptr),
        ctypes.c_void_p(out_ptr)
    )

    # 7. Assert precision math up to standard float32 tolerance margins
    cpp_output_pt = torch.from_numpy(cpp_output_np)
    is_correct = torch.allclose(golden_output, cpp_output_pt, rtol=1e-4, atol=1e-4)

    if is_correct:
        print(" SUCCESS: Custom compiled kernel output matches PyTorch perfectly!")
        max_diff = torch.max(torch.abs(golden_output - cpp_output_pt)).item()
        print(f"Maximum absolute numerical drift: {max_diff:.6e}")
    else:
        print(" ERROR: Numerical mismatch detected between compiled kernel and PyTorch reference.")