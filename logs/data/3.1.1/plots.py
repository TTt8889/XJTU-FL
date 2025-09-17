import matplotlib.pyplot as plt
import numpy as np

# 设置字体大小为五号字体（大约9pt）
plt.rcParams.update({'font.size': 15})

# 模拟数据
# alie_acc = np.load(f"acc_MNIST_ALIE_TrimmedMean_2025-02-26_14-12-39.npy")
# alie_sign_flip = np.load(f"sign-flip_MNIST_ALIE_TrimmedMean.npy")
# alie_sign_flip_s = [v * 100 for v in alie_sign_flip[1]]

# ipm_acc = np.load(f"acc_MNIST_IPM_TrimmedMean_2025-02-26_14-13-00.npy")
# ipm_sign_flip = np.load(f"sign-flip_MNIST_IPM_TrimmedMean.npy")
# ipm_sign_flip_s = [v * 100 for v in ipm_sign_flip[1]]

# gauss_acc = np.load(f"acc_MNIST_Gaussian_TrimmedMean_2025-02-26_14-34-05.npy")
# gauss_sign_flip = np.load(f"sign-flip_MNIST_Gaussian_TrimmedMean.npy")
# gauss_sign_flip_s = [v * 100 for v in gauss_sign_flip[1]]

# fang_acc = np.load(f"acc_MNIST_FangAttack_TrimmedMean_2025-02-26_14-34-14.npy")
# fang_sign_flip = np.load(f"sign-flip_MNIST_FangAttack_TrimmedMean.npy")
# fang_sign_flip_s = [v * 100 for v in fang_sign_flip[1]]

# no_attack_acc = np.load(f"acc_MNIST_NoAttack_TrimmedMean_2025-02-26_14-12-13.npy")
# no_attack_sign_flip = np.load(f"sign-flip_MNIST_NoAttack_TrimmedMean.npy")
# no_attack_sign_flip_s = [v * 100 for v in no_attack_sign_flip[1]]

# mpaf_acc = np.load(f"acc_MNIST_MPAF_TrimmedMean_2025-02-26_14-57-00.npy")
# mpaf_sign_flip = np.load(f"sign-flip_MNIST_MPAF_TrimmedMean.npy")
# mpaf_sign_flip_s = [v * 100 for v in mpaf_sign_flip[1]]

# # 2025/9/16新增MyAttack
# myattack_acc = np.load(f"acc_MNIST_MyAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-13-57.npy")
# myattack_sign_flip = np.load(f"sign-flip_MNIST_MyAttack_TrimmedMean_iid-0.2_adv-0.2_2025-09-16_22-31-59.npy")
# myattack_sign_flip_s = [v * 100 for v in myattack_sign_flip[1]]

alie_acc = np.load(f"acc_MNIST_ALIE_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-39.npy")
alie_sign_flip = np.load(f"sign-flip_MNIST_ALIE_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-39.npy")
alie_sign_flip_s = [v * 100 for v in alie_sign_flip[1]]

ipm_acc = np.load(f"acc_MNIST_IPM_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-50.npy")
ipm_sign_flip = np.load(f"sign-flip_MNIST_IPM_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-50.npy")
ipm_sign_flip_s = [v * 100 for v in ipm_sign_flip[1]]

gauss_acc = np.load(f"acc_MNIST_Gaussian_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-09.npy")
gauss_sign_flip = np.load(f"sign-flip_MNIST_Gaussian_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-09.npy")
gauss_sign_flip_s = [v * 100 for v in gauss_sign_flip[1]]

fang_acc = np.load(f"acc_MNIST_FangAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-58.npy")
fang_sign_flip = np.load(f"sign-flip_MNIST_FangAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-58.npy")
fang_sign_flip_s = [v * 100 for v in fang_sign_flip[1]]

no_attack_acc = np.load(f"acc_MNIST_NoAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-39-57.npy")
no_attack_sign_flip = np.load(f"sign-flip_MNIST_NoAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-39-57.npy")
no_attack_sign_flip_s = [v * 100 for v in no_attack_sign_flip[1]]

mpaf_acc = np.load(f"acc_MNIST_MPAF_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-42.npy")
mpaf_sign_flip = np.load(f"sign-flip_MNIST_MPAF_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-42.npy")
mpaf_sign_flip_s = [v * 100 for v in mpaf_sign_flip[1]]

myattack_acc = np.load(f"acc_MNIST_MyAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-39.npy")
myattack_sign_flip = np.load(f"sign-flip_MNIST_MyAttack_Mean_iid-0.2_adv-0.2_2025-09-17_14-40-39.npy")
myattack_sign_flip_s = [v * 100 for v in myattack_sign_flip[1]]


# 创建图形和子图
fig1 = plt.figure(figsize=(6, 5))  # 将厘米转换为英寸 
ax1 = fig1.add_subplot(111)

# 左图：测试错误率
ax1.plot(gauss_acc[0], gauss_acc[1], 'r--', label='Gaussian', linewidth=3)
ax1.plot(fang_acc[0], fang_acc[1], 'k', label='Fang', linewidth=3)
ax1.plot(myattack_acc[0], myattack_acc[1], 'y', label='MyAttack', linewidth=3)
ax1.plot(no_attack_acc[0], no_attack_acc[1], 'g', label='No Attack', linewidth=3)
ax1.plot(ipm_acc[0], ipm_acc[1], 'b--', label='IPM', linewidth=3)
ax1.plot(alie_acc[0], alie_acc[1], 'm', label='ALIE', linewidth=3)
ax1.plot(mpaf_acc[0], mpaf_acc[1], 'c', label='MPAF', linewidth=3)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Test accuracy (%)')
ax1.set_ylim(0, 100)
ax1.legend()
ax1.grid(True)

fig1.savefig('test_acc_nodefense.svg', format='svg', bbox_inches='tight', dpi=300) 


fig2 = plt.figure(figsize=(6, 5))  # 将厘米转换为英寸 
ax2 = fig2.add_subplot(111)
# 右图：翻转率
ax2.plot(gauss_sign_flip[0], gauss_sign_flip_s, 'r--', label='Gaussian', linewidth=3)
ax2.plot(fang_sign_flip[0], fang_sign_flip_s, 'k', label='Fang', linewidth=3)
ax2.plot(myattack_sign_flip[0], myattack_sign_flip_s, 'y', label='MyAttack', linewidth=3)
ax2.plot(ipm_sign_flip[0], ipm_sign_flip_s, 'b--', label='IPM', linewidth=3)
ax2.plot(mpaf_sign_flip[0], mpaf_sign_flip_s, 'c', label='MPAF', linewidth=3)
ax2.plot(alie_sign_flip[0], alie_sign_flip_s, 'm', label='ALIE', linewidth=3)
ax2.plot(no_attack_sign_flip[0], no_attack_sign_flip_s, 'g', label='No Attack', linewidth=3)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Flip Rate (%)')
ax2.set_ylim(0, 100)
ax2.legend()
ax2.grid(True)

fig2.savefig('flip_rate_nodefense.svg', format='svg', bbox_inches='tight', dpi=300)


# 显示图形
plt.show()