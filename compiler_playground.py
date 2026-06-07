# compiler_playground.py
import sys
from llvmlite import ir

def walk_through_llvm_cpp_api():
    print("\n==============================================================")
    # Traditional systems arechitecture uses factory builders to create blocks
    print("--- [ERA 2] Programmatic LLVM C++ API Generation (llvmlite) ---")
    print("================================================================")

    # 1. Intialize a formal Module container object
    module = ir.Module(name="microtensor_cpp_api_module")

    # 2. Define function signature: void microtensor_kernel(float* t1, float* t2, float* output)
    float_ptr = ir.PointerType(ir.FloatType())
    func_type = ir.FunctionType(ir.VoidType(), [float_ptr, float_ptr, float_ptr])
    func = ir.Function(module, func_type, name="microtensor_kernel")

    # 3. Create basic blocks and anchor an IRBuilder to it
    entry_block = func.append_basic_block(name="entry")
    builder = ir.IRBuilder(entry_block)

    # 4. Extract arguments
    t1_ptr, t2_ptr, out_ptr = func.args

    # 5. Programmatically build instructions (Type safety is enforced by the objects!)
    val1 = builder.load(t1_ptr, name="val1")
    val2 = builder.load(t2_ptr, name="val2")

    # Perform element-wise multiply
    mul_result = builder.fmul(val1, val2, name="mul_tmp")

    # Store out to our destionation pointer
    builder.store(mul_result, out_ptr)
    builder.ret_void()

    # 6. Dump the generated text string
    print("[SUCCESS] Factory builder generated clean, type-safe LLVM assembly:")
    print("-------------------------------------------------------------------")
    print(str(module).strip())
    print("-------------------------------------------------------------------")



def walk_through_mlir_dialect_api():
    print("\n=================================================================")
    print("--- [ERA 3] Modern MLIR Graph Processing Lowering ---")
    print("===================================================================")

    print("Industrial AI compilers do not jump from a graph straight to assembly.")
    print("Instead, they lower through specialized 'Dialects' layer-by-layer.\n")

    # Let's trace how a Matrix Multiply transforms through the compilation pipeline:

    # --- LEVEL 1: High-Level Graph Dialect (e.g., TOSA/ StableHLO) ---
    print("STAGE 1: High-Level Graph Dialect (Keeps Tensor Sematics)")
    tosa_dialect = (
        "func.func @forward(%t1: tensor<128x512xf32>, %t2: tensor<512x256xf32>) -> tensor<128x256xf32> {\n}"
        "   %0 = tosa.matmul %t1, %t2 : (tensor<128x512xf32>, tensor<512x256xf32>) -> tensor<128x256xf32>\n"
        "   return %0 : tensor<128x256xf32>\n"
        "}"
    )
    print(tosa_dialect)
    print("-" * 56)

    # --- LEVEL 2: Mid-Level Structured Loop Dialect (e.g., Linalg / Affine) ---
    print("STAGE 2: Lowering to Linalg/Affine Dialect (Computes Loop Tiling & Strides)")
    linalg_dialect = (
        "affine.for %i = 0 to 128 {\n"
        "   affine.for %j = 0 to 256 {\n"
        "       affine.for %k = 0 to 512 {\n"
        "           // Compiler optimizes memory blocks, cache locality, and threads here!\n"
        "           %acc = ...\n"
        "       }\n"
        "   }\n"
        "}"
    )
    print(linalg_dialect)
    print("-" * 56)

    # --- LEVEL 3: Low-Level Instruction Dialect (Scalar LLVM IR Map) ---
    print("STAGE 3: Final Lowering Pass (Emits the raw scalar LLVM IR we compiled!)")
    print("     %ptr1 = getelementptr inbounds float, float* %t1, i32 %idx")
    print("     %val1 = load float, float* %ptr1, align 4")
    print("     %mul = fmul float %val1, %val2")
    print("=============================================================")


if __name__ == "__main__":
    walk_through_llvm_cpp_api()
    walk_through_mlir_dialect_api()