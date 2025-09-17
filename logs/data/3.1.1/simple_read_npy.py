# simple_read_npy.py：极简通用.npy文件读取工具
import numpy as np
import os
# import pandas as pd 

def read_any_npy(file_path):
    # 1. 基础检查
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在：{file_path}")
        return
    if not file_path.endswith(".npy"):
        print(f"❌ 不是.npy文件：{file_path}")
        return

    # 2. 读取文件（自动适配所有数组类型）
    try:
        data = np.load(file_path)
        # df = pd.DataFrame(data)
    except Exception as e:
        # 若包含Python对象，提示开启allow_pickle
        if "allow_pickle" in str(e):
            data = np.load(file_path, allow_pickle=True)
            print("⚠️  已开启pickle模式（仅读取可信文件！）")
        else:
            print(f"❌ 读取失败：{e}")
            return

    # 3. 显示核心信息（所有.npy文件通用）
    print(f"✅ 读取成功！")
    print(f"📏 数据形状：{data.shape}  |  数据类型：{data.dtype}")
    print(f"📊 数值范围：{data.min():.4f} ~ {data.max():.4f}")
    print(data)
    # df.head(10)  # 显示前10行

# --------------------------
# 唯一需要改的地方：文件路径
# --------------------------
if __name__ == "__main__":
    # 替换为你的.npy文件路径（相对/绝对路径都可以）
    npy_path = "/home/ymh/wxy/Sign-Fed/logs/data/3.1.1/acc_MNIST_MyAttack_Mean_iid-0.2_adv-0.2_2025-09-17_15-16-43.npy"
    # npy_path = "/home/ymh/wxy/Sign-Fed/logs/data/3.1.1/sign-flip_MNIST_MyAttack_TrimmedMean_iid-0.2_adv-0.2_2025-09-16_22-31-59.npy"
    # npy_path = "/home/ymh/wxy/Sign-Fed/logs/data/3.1.1/acc_MNIST_MPAF_TrimmedMean_2025-02-26_14-57-00.npy"
    # npy_path = "./ZTask_Log/acc_3.1.1.npy"          # 示例：准确率数据
    # npy_path = "/home/ymh/wxy/Sign-Fed/xxx.npy"     # 示例：绝对路径（更稳妥）
    
    # 执行读取
    read_any_npy(npy_path)