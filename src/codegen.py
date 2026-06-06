# src/codegen.py

class LoopCodegen:
    @staticmethod
    def generate(graph, offsets, total_pool_size) -> str:
        """
        Translates the high-level tensor graph and static memory plan
        into a highly optimized, fused C++ execution kernel.
        """
        cpp_code = []

        # 1. Header and Standard Boilerplate
        cpp_code.append("#include <iostream>")
        cpp_code.append("#include <cmath>")
        cpp_code.append("#include <cstdint>\n")

        cpp_code.append(f"// Allocating global static scratchpad memory pool")
        cpp_code.append(f"uint8_t memory_pool[{total_pool_size}];\n")

        cpp_code.append("// Fused Microtensr Inference Kernel")
        cpp_code.append('extern "C" __declspec(dllexport) void microtensor_kernel(const float* t1, const float* t2, const float* t3, float* output) {')

        # 2. Reconstruct pointer offsets from our allocator plan
        cpp_code.append("// Reconstructed buffer aliases from static memory plan")
        # In our graph, %t6 is the final output layer, but let's bind our internal buffers
        cpp_code.append(f" float* t4 = reinterpret_cast<float*>(&memory_pool[{offsets['t4']}]);")
        cpp_code.append(f" float* t5 = reinterpret_cast<float*>(&memory_pool[{offsets['t5']}]);")
        cpp_code.append(f" float* t6 = reinterpret_cast<float*>(&memory_pool[{offsets['t6']}]);\n")

        cpp_code.append("// FUSED LOOP NEST: MatMul + Add + ReLU")
        cpp_code.append(" for (int i = 0; i< 128; ++i) {")
        cpp_code.append("   for (int j = 0; j< 256; ++j) {")
        cpp_code.append("       // --- Phase A: Compute Matmul Accumulation ---")
        cpp_code.append("       float acc = 0.0f;")
        cpp_code.append("       for (int k = 0; k < 512; ++k) {")
        cpp_code.append("           acc += t1[i * 512 + k] * t2[k * 256 + j];")
        cpp_code.append("       }")
        cpp_code.append("       ")
        cpp_code.append("       // --- Phase B: Inline Operator Fusion (Elementwise Add) ---")
        cpp_code.append("       float val_add = acc + t3[i * 256 + j];")
        cpp_code.append("       ")
        cpp_code.append("       // --- Phase C: Inline Operator Fusion (Activation ReLU) ---")
        cpp_code.append("       float val_relu = (val_add > 0.0f) ? val_add : 0.0f;")
        cpp_code.append("       ")
        cpp_code.append("       // Stream directly out to our optimized destination pointer window")
        cpp_code.append("       t6[i * 256 + j] = val_relu;")
        cpp_code.append("       }")
        cpp_code.append("   }")

        # 3. Copy final output layer buffer out to user's destination array
        cpp_code.append("\n // Copy internal output back to host destination pointer")
        cpp_code.append("   for (int idx = 0; idx < 128 * 256; ++idx) {")
        cpp_code.append("       output[idx] = t6[idx];")
        cpp_code.append("   }")

        cpp_code.append("}")

        return "\n".join(cpp_code)

    def generate_llvm_ir(size=2048):
        llvm_lines = [
            "; ModuleID = 'microtensor_kernel'",
            'target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n81:128-a:0:64-S128"',
            'target triple = "x86_64-pc-windows-gnu"',
            "",
            '; Define our exported FFI function symbol',
            f'define void @microtensor_kernel(float* noalias %t1, float* noalias %t2, float* nonalias %t3, float* nonalias %output) {{',
            "entry:",
            "   ; Jump straight into our loop block",
            "   br label %loop.body",
            "",
            "loop.body:",
            "   ; Define our loop index variable using a Phi node or a simple counter register",
            f"  ; For simplicity, let's look at how a single unrolled or hardcoded step looks:",
            "   %idx = phi i32 [0, %entry], [ %next_idx, %loop.body ]",
            "",
            "   ; 1. Calculate pointer offsets via GetElementPtr(GEP)",
            "   %ptr1 = getelementptr inbounds float, float* %t1, i32 %idx",
            "   %ptr2 = getelementptr inbounds float, float* %t2, i32 %idx",
            "   %ptr3 = getelementptr inbounds float, float* %t3, i32 %idx",
            "   %ptr_out = getelementptr inbounds float, float* %output, i32 %idx",
            "",
            "   ; 2. Load the literal values out of memory from those calculated addresses",
            "   %val1 = load float, float* %ptr1, align 4",
            "   %val2 = load float, float* %ptr2, align 4",
            "   %val3 = load float, float* %ptr3, align 4",
            "",
            "   ; 3. Perform the fused element-wise Math (val1 * val2 + val3)",
            "   %mul = fmul float %val1, %val2",
            "   %add = fadd float %mul, %val3",
            "",
            "   ; 4. Apply manual Max(0, x) for the Fused ReLU activation pass",
            "   %cmp = fcmp ogt float %add, 0.000000e+00",
            "   %relu = select i1 %cmp, float %add, float 0.000000e+00",
            "",
            "   ; 5. Store the final result back to the output array pointer",
            "   store float %relu, float* %ptr_out, align 4",
            "",
            "   ; 6. Loop Control logic",
            "   %next_idx = add nuw nsw i32 %idx, 1",
            f"  %loop_cond = icmp ult i32 %next_idx, {size}",
            "   br i1 %loop_cond, label %loop.body, label %loop.exit",
            "",
            "loop.exit:",
            "   ret void",
            "}"
        ]
        return "\n".join(llvm_lines)

        ]
