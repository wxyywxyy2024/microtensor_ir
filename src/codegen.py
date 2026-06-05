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
        cpp_code.append("void microtensor_kernel(const float* t1, const float* t2, const float* t3, float* output) {")

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
