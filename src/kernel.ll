; ModuleID = 'microtensor_kernel'
target datalayout = "e-m:w-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n81:128-a:0:64-S128"
target triple = "x86_64-pc-windows-gnu"

; Define our exported FFI function symbol
define void @microtensor_kernel(float* noalias %t1, float* noalias %t2, float* noalias %t3, float* noalias %output) {
entry:
   br label %loop.i.head

; --- OUTER LOOP: i = 0 to 128 ---
loop.i.head:
   %i = phi i32 [0, %entry], [ %i.next, %loop.i.latch]
   %i.stride.t1 = mul i32 %i, 512
   %i.stride.out = mul i32 %i, 256
   br label %loop.j.head

; ---MIDDLE LOOP: j = 0 to 256 ---
loop.j.head:
   %j = phi i32 [0, %loop.i.head], [%j.next, %loop.j.latch]
   br label %loop.k.head

; --- INNER LOOP HEAD: k = 0 to 512 (Accumulation) ---
loop.k.head:
   %k = phi i32 [0, %loop.j.head], [%k.next, %loop.k.body]
   %acc.in = phi float [ 0.000000e+00, %loop.j.head], [%acc.next, %loop.k.body]
   br label %loop.k.body

loop.k.body:
   ; t1[i * 512 + k]
   %idx.t1 = add i32 %i.stride.t1, %k
   %ptr.t1 = getelementptr inbounds float, float* %t1, i32 %idx.t1
   %val.t1 = load float, float* %ptr.t1, align 4

   ; t2[k * 256 + j]
   %k.stride.t2 = mul i32 %k, 256
   %idx.t2 = add i32 %k.stride.t2, %j
   %ptr.t2 = getelementptr inbounds float, float* %t2, i32 %idx.t2
   %val.t2 = load float, float* %ptr.t2, align 4

   ; acc += t1 * t2
   %mul = fmul float %val.t1, %val.t2
   %acc.next = fadd float %acc.in, %mul

   %k.next = add nuw nsw i32 %k, 1
   %k.cond = icmp ult i32 %k.next, 512
   br i1 %k.cond, label %loop.k.head, label %loop.k.exit

; --- INNER LOOP EXIT: Elementwise Fusion (Bias Add + ReLU) ---
loop.k.exit:
   ; t3[i * 256 + j]
   %idx.bias =  add i32 %i.stride.out, %j
   %ptr.bias = getelementptr inbounds float, float* %t3, i32 %idx.bias
   %val.bias = load float, float* %ptr.bias, align 4

   ; Add bias
   %val.add = fadd float %acc.next, %val.bias

   ; ReLU: max(0.0. val_add)
   %cmp = fcmp ogt float %val.add, 0.000000e+00
   %val.relu = select i1 %cmp, float %val.add, float 0.000000e+00

   ; Store out to destination pointer window: output[i * 256 + j]
   %ptr.out = getelementptr inbounds float, float* %output, i32 %idx.bias
   store float %val.relu, float* %ptr.out, align 4
   br label %loop.j.latch

; --- MIDDLE LOOP LATCH ---
loop.j.latch:
   %j.next = add nuw nsw i32 %j, 1
   %j.cond = icmp ult i32 %j.next, 256
   br i1 %j.cond, label %loop.j.head, label %loop.i.latch

; ---OUTER LOOP LATCH---
loop.i.latch:
   %i.next = add nuw nsw i32 %i, 1
   %i.cond = icmp ult i32 %i.next, 128
   br i1 %i.cond, label %loop.i.head, label %root.exit

root.exit:
   ret void
}