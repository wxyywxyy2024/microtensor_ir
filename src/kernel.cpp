#include <iostream>
#include <cmath>
#include <cstdint>

// Allocating global static scratchpad memory pool
uint8_t memory_pool[262144];

// Fused Microtensr Inference Kernel
void microtensor_kernel(const float* t1, const float* t2, const float* t3, float* output) {
// Reconstructed buffer aliases from static memory plan
 float* t4 = reinterpret_cast<float*>(&memory_pool[0]);
 float* t5 = reinterpret_cast<float*>(&memory_pool[131072]);
 float* t6 = reinterpret_cast<float*>(&memory_pool[0]);

// FUSED LOOP NEST: MatMul + Add + ReLU
 for (int i = 0; i< 128; ++i) {
   for (int j = 0; j< 256; ++j) {
       // --- Phase A: Compute Matmul Accumulation ---
       float acc = 0.0f;
       for (int k = 0; k < 512; ++k) {
           acc += t1[i * 512 + k] * t2[k * 256 + j];
       }
       
       // --- Phase B: Inline Operator Fusion (Elementwise Add) ---
       float val_add = acc + t3[i * 256 + j];
       
       // --- Phase C: Inline Operator Fusion (Activation ReLU) ---
       float val_relu = (val_add > 0.0f) ? val_add : 0.0f;
       
       // Stream directly out to our optimized destination pointer window
       t6[i * 256 + j] = val_relu;
       }
   }

 // Copy internal output back to host destination pointer
   for (int idx = 0; idx < 128 * 256; ++idx) {
       output[idx] = t6[idx];
   }
}