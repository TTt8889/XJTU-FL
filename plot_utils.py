import re
import matplotlib.pyplot as plt
import numpy as np
import datetime


def parse_logs(filename):
    plt.clf()
    # read log file
    with open(filename, 'r') as f:
        content = f.read()
    epochs, accs, losses, asrs, asr_losses = [], [], [], [], []
    # regular expression pattern to extract epoch, test accuracy, test loss, asr, asr loss
    regex = (
        r"Epoch (?P<epoch>\d+)\s.*?Test Acc: (?P<test_acc>[\d\.]+)\s.*?Test loss: (?P<test_loss>[\d\.]+)(?:\s.*?ASR: (?P<asr>[\d\.]+))?(?:\s.*?ASR loss: (?P<asr_loss>[\d\.]+))?"
    )

    for match in re.finditer(regex, content):
        epochs.append(int(match.group('epoch')))
        accs.append(float(match.group('test_acc')))
        losses.append(float(match.group('test_loss')))

        # if asr and asr loss exist, add them, or add None
        asr = match.group('asr')
        asr_loss = match.group('asr_loss')
        asrs.append(float(asr) if asr else None)
        asr_losses.append(float(asr_loss) if asr_loss else None)

    return epochs, accs, losses, asrs, asr_losses



def plot_label_distribution(train_data, client_idcs, n_clients, dataset, distribution, dirichlet_alpha):
    titleid_dict = {"iid": "Balanced_IID", "class-imbalanced_iid": "Class-imbalanced_IID",
                    "non-iid": "Quantity-imbalanced_Dirichlet_Non-IID", "pat": "Balanced_Pathological_Non-IID", "imbalanced_pat": "Quantity-imbalanced_Pathological_Non-IID"}
    dataset = "CIFAR-10" if dataset == "CIFAR10" else dataset
    title_id = dataset + " " + titleid_dict[distribution]
    xy_type = "client_label"  # 'label_client'
    plt.rcParams['font.size'] = 15  # set global fontsize
    # set the direction of xtick toward inside
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['font.sans-serif'] = ['Times New Roman']
    labels = train_data.targets
    n_classes = labels.max()+1
    plt.figure(figsize=(12, 8))
    if xy_type == "client_label":
        label_distribution = [[] for _ in range(n_classes)]
        for c_id, idc in enumerate(client_idcs):
            for idx in idc:
                label_distribution[labels[idx]].append(c_id)

        plt.hist(label_distribution, stacked=True,
                 bins=np.arange(-0.5, n_clients + 1.5, 1),
                 label=range(n_classes), rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_clients), ["%d" %
                                          c_id for c_id in range(n_clients)])
        plt.xlabel("Client ID", fontsize=20)
    elif xy_type == "label_client":
        plt.hist([labels[idc]for idc in client_idcs], stacked=True,
                 bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
                 label=["Client {}".format(i) for i in range(n_clients)],
                 rwidth=0.5, zorder=10)
        plt.xticks(np.arange(n_classes), train_data.classes)
        plt.xlabel("Label type", fontsize=20)

    plt.ylabel("Number of Training Samples", fontsize=20)
    # plt.title(f"{title_id} Label Distribution Across Clients", fontsize=20)
    rotation_degree = 45 if n_clients > 30 else 0
    plt.xticks(rotation=rotation_degree, fontsize=16)
    plt.legend(loc="best", prop={'size': 12}).set_zorder(100)
    plt.grid(linestyle='--', axis='y', zorder=0)
    plt.tight_layout()
    # plt.savefig(f"./logs/{title_id}_label_dtb.pdf",
    #             format='pdf', bbox_inches='tight')
    plt.savefig(f"./logs/{dirichlet_alpha}_{title_id}_label_dtb.svg", 
                format='svg', bbox_inches='tight', dpi=300)

def plot_fscores(args, data, dpi=300):
    """
    绘制三维散点图并保存到文件。

    参数:
    - data: 输入数据，格式为 [[[y1_z1, z1], [y2_z1, z2]], ...]
            其中每个子列表代表一个 epoch，包含两个客户端的数据。
            每个客户端的数据是一个包含两个值的列表，分别作为 y 轴和 z 轴的值。
    - dpi: 图像的分辨率，默认为 300。
    """
    # 提取 x, y, z 数据
    x1, y1, z1 = [], [], []  # 第一个客户端
    x2, y2, z2 = [], [], []  # 第二个客户端

    # 遍历每个 epoch
    for epoch, clients in enumerate(data, start=args.predict_epochs):
        # 第一个客户端
        x1.append(epoch)
        y1.append(clients[0][0])
        z1.append(clients[0][1])

        # 第二个客户端
        x2.append(epoch)
        y2.append(clients[1][0])
        z2.append(clients[1][1])

    # 保存在npy，至于怎么用，就在我的电脑上写代码了，这里的画图只是随便看一下
    save_npy_date(args, [[x1, y1, z1], [x2, y2, z2]], args.npy_file_name, "flip-score")

    # 创建 3D 图
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # 绘制第一个客户端的数据
    scatter1 = ax.scatter(x1, y1, z1, c='red', label='Malicious Client')  # 添加 label 参数
    # 绘制第二个客户端的数据
    scatter2 = ax.scatter(x2, y2, z2, c='green', label='Benign Client')  # 添加 label 参数

    # 设置坐标轴标签
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Old flip scores')
    ax.set_zlabel('Pred flip scores')

    # 设置图像标题
    ax.set_title(f'{args.defense} with {args.attack}')

    # 添加图例
    ax.legend()  # 自动从 label 参数提取图例

    # 保存图像到文件
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'/home/ymh/wxy/Sign-Fed/logs/Flip_score_picture/3d_{args.dataset}_{args.attack}_{args.defense}_{args.distribution}-{args.dirichlet_alpha}_adv-{args.num_adv/args.num_clients}_{formatted_time}.png'
    plt.savefig(filename, dpi=dpi)
    plt.close()

def plot_accuracy(args):
    epochs, accs, _, asr, _ = parse_logs(args.output)
    accs_scaled = [value * 100 for value in accs] # 换成百分数

    if args.my_exp:
        # 这里保存acc
        save_npy_date(args, [epochs, accs_scaled], args.npy_file_name, "acc")

    plt.plot(epochs, accs_scaled, label='Accuracy')

    # if asr statistics exists, plot asr curve
    if any(asr):
        plt.plot(epochs, asr, label='ASR', linestyle='--')

    plt.xlabel('Epoch')
    plt.ylabel('Test accuracy (%)')
    # plt.title('Accuracy vs Epoch')
    plt.legend()
    plt.grid(True)
    plt.savefig(args.output[:-4] + ".png")
    plt.close()


def save_npy_date(args, all_data, npy_file_name, npy_prefix):
    # 记得连epoch一起保存，[epoch_list, data_list]
    assert npy_prefix is not None, "npy_prefix can be None"
    array_data = np.array(all_data)
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.save(f"/home/ymh/wxy/Sign-Fed/logs/data/{npy_file_name}/{npy_prefix}_{args.dataset}_{args.attack}_{args.defense}_{args.distribution}-{args.dirichlet_alpha}_adv-{args.num_adv/args.num_clients}_{formatted_time}.npy", array_data)


if __name__ == "__main__":
    pass
