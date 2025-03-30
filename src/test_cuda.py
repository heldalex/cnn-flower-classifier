import torch


print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())  # Should return True
print(torch.cuda.get_device_name(0))  # Should show RTX 4070 Ti
