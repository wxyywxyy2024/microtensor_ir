# test_run.py
import torch
from src.fx_frontend import import_torch_model
from src.optimizer import LivenessAnalyzer

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



