import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Version: {torch.version.cuda}")
print(f"Is CUDA Available: {torch.cuda.is_available()}")

a = torch.randn(3, 3).cuda()
print(a)