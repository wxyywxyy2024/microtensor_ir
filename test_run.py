# test_run.py
import torch
from src.fx_frontend import import_torch_model
from src.optimizer import LivenessAnalyzer
from src.optimizer import StaticMemoryAllocator

# 1. Define a native PyTorch block
class TinyTransformerBlock(torch.nn.Module):
    def forward(self, x, weights, bias):
        # A simple linear projection layer followed by a ReLU activation
        h1 = torch.mm(x, weights)
        h2 = h1 + bias
        return torch.relu(h2)

# 2. Setup mock data dimensions matching our shape expectations
x_mock = torch.randn(128, 512)
w_mock = torch.randn(512, 256)
b_mock = torch.randn(128, 256)

model = TinyTransformerBlock()

# 3. Run our compiler frontend ingestion pipeline!
print("--- [Step 1] Ingesting PyTorch Model via torch.fx ---")
compiler_graph = import_torch_model(model, (x_mock, w_mock, b_mock))
compiler_graph.print_ir()

print("\n--- [Step 2] Executing Middle-End Liveness Analysis ---")
intervals = LivenessAnalyzer.analyze(compiler_graph)

print(f"{'Tensor Name':<15} | {'Birth Line':<12} | {'Death Line':<12}")
print("-" * 47)
for tensor_name, (birth, death) in intervals.items():
    print(f"%{tensor_name:<14} | {birth:<12} | {death:<12}")

print("\n--- [Step 3] Running Static Memory Allocator Optimization ---")
offsets, total_pool_size = StaticMemoryAllocator.allocate(compiler_graph)

print(f"{'Tensor Buffer':<15} | {'Assigned Byte Offset':<22} | {'Size (KB)':<10}")
print("-" * 55)
native_total_size = 0
for t_name, offset in offsets.items():
    # Fetch tensor object to calculate its individual size
    node = next(n for n in compiler_graph.nodes if n.outputs[0].name == t_name)
    size_kb = StaticMemoryAllocator.calculate_bytes(node.outputs[0].shape, "float32")/1024
    native_total_size += size_kb
    print(f"%{t_name:<14} | {hex(offset):<22} | {size_kb:<10} KB")

print("-" * 55)
print(f"Native Allocation Total Size : {native_total_size} KB")
print(f"Optimized Managed Pool Size  : {total_pool_size / 1024} KB")
print(f"Hardware Memory Saved        : {native_total_size - (total_pool_size / 1024)} KB")



