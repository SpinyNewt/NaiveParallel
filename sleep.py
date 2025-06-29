import torch
import random

# 目标显存：4GB（以字节为单位）
target_bytes = 4 * 1024**3  # 4GB

# 计算所需元素数量
element_size = 4  # float32
n_elements = target_bytes // element_size

# 创建张量
r = torch.zeros(n_elements, dtype=torch.float32).cuda()

# 验证显存占用
print(f"分配的显存: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

import time
sleep_time = random.randint(20, 50)  # 随机休眠时间
time.sleep(sleep_time)
print(f"休眠 {sleep_time} 秒,已完成")
del r