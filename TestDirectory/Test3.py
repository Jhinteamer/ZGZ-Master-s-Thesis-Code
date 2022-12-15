import torch
print(torch.__version__)
print(torch.cuda.is_available())

print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号

import Os_test.Test4
