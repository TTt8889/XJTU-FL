import numpy as np
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client
from fl.models import get_model
from fl.models.model_utils import model2vec
from copy import deepcopy
from scipy.stats import binom
import datetime
from plot_utils import save_npy_date


@attacker_registry
@actor('attacker', 'model_poisoning', 'non_omniscient')
class MyAttack(MPBase, Client):

    # 注意会有多个攻击者客户端，所以要用静态变量
    target_sign = None
    s1 = None
    s2 = None
    # 上一次的全局模型，和预测的全局模型
    last_weight_vec = None
    predict_weight_vec = None
    # 多个攻击者每轮次只有一次修改标识
    already_modified = False
    last_update = None
    now_update = None

    # 动态缩放系数
    lamada = 8
    decay = 0.75
    e = 0
    hyptest_model = None
    first_record = True

    # 牺牲客户端
    K = 10
    aff_update = None

    # 动量系数
    gamma = 0.5
    last_moment_update = None

    updata_sign_match_list = []


    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {}
        self.alpha = args.predict_alpha
        self.update_and_set_attr()

        # 初始化静态变量
        if MyAttack.target_sign is None:
            # 这里的正负比例也是可控的，加一个p=[0.3, 0.7]
            model_shape = model2vec(deepcopy(get_model(args))).shape
            MyAttack.target_sign = np.random.choice([-1, 1], size=model_shape)
            MyAttack.print_target_sign(MyAttack.target_sign)

            MyAttack.s1 = np.zeros(shape=model_shape)
            MyAttack.s2 = np.zeros_like(MyAttack.s1)

    def non_omniscient(self):
        # MyAttack.print_target_sign(MyAttack.target_sign)
        if not MyAttack.already_modified:
            MyAttack.already_modified = True

            # 为了初始化s的预热轮次，先传目标符号的随机噪声
            if self.global_epoch < self.args.predict_epochs:
                MyAttack.now_update = MyAttack.target_sign * np.abs(np.random.normal(
                    loc=0.0, scale=1.0, size=self.update.shape).astype(np.float32))
                
                MyAttack.s1 += self.global_weights_vec * (1 / self.args.predict_epochs)
                MyAttack.s2 += self.global_weights_vec * (1 / self.args.predict_epochs)

            # 进入攻击轮次
            else:
                MyAttack.s1 = self.alpha * self.global_weights_vec + (1 - self.alpha) * MyAttack.s1
                MyAttack.s2 = self.alpha * MyAttack.s1 + (1 - self.alpha) * MyAttack.s2
                MyAttack.predict_weight_vec = (2 * MyAttack.s1 - MyAttack.s2 ) + (self.alpha / (1 - self.alpha)) * ( MyAttack.s1 - MyAttack.s2 )

                # global_weight_update = self.global_weights_vec - MyAttack.last_weight_vec
                # global_weight_update_norm = np.linalg.norm(global_weight_update)
                # last_update_norm = np.linalg.norm(MyAttack.last_update)

                # v = global_weight_update - (global_weight_update_norm/last_update_norm)*MyAttack.last_update
                # uni_bengin_update = v / np.linalg.norm(v)
                # v_s = MyAttack.target_sign * np.abs(uni_bengin_update)

                # 优化一下恶意更新的模长，使其不容易被防御住
                # pre_norm = np.linalg.norm(MyAttack.predict_weight_vec - self.global_weights_vec)
                # base_norm = (pre_norm + global_weight_update_norm) / 2
                
                # 每e轮次间隔，假设检验lamda
                if (MyAttack.e % self.args.hyper_e == 0):
                    if MyAttack.first_record:
                        MyAttack.hyptest_model = self.global_weights_vec
                        MyAttack.first_record = False
                    else:
                        param_size = len(self.global_weights_vec)
                        param_match = np.sum(MyAttack.target_sign == np.sign(self.global_weights_vec - MyAttack.hyptest_model))
                        print(f"Adjust Lamada, param match: {param_match}")
                        if not MyAttack.should_reject_H0(param_size, param_match, 0.01):
                            MyAttack.lamada = max(0.5, MyAttack.lamada * MyAttack.decay)
                            print(f"Now Lamada: {MyAttack.lamada}")
                        MyAttack.hyptest_model = self.global_weights_vec
                
                predict_weight_update = MyAttack.predict_weight_vec - self.global_weights_vec
                predict_grad_update = predict_weight_update - (self.global_weights_vec - MyAttack.last_weight_vec)
                
                predict_grad_update_norm = np.linalg.norm(predict_grad_update)
                predict_grad_update_uni = predict_grad_update / predict_grad_update_norm

                v_s = predict_weight_update + ((MyAttack.target_sign * np.abs(predict_grad_update_uni)) * MyAttack.lamada)

                # 是否开启动量
                if self.moment_open:
                    if MyAttack.last_moment_update is None:
                        MyAttack.last_moment_update = v_s
                    else:
                        v_s = MyAttack.gamma * MyAttack.last_moment_update + (1-MyAttack.gamma) * v_s
                        MyAttack.last_moment_update = v_s
                    

                # 是否开启牺牲客户端，没开启就用一样的
                if self.aff_open:
                    # 这里就感觉怪怪的，先用这个吧
                    MyAttack.aff_update = -(predict_weight_update + ((MyAttack.target_sign * np.abs(predict_grad_update_uni)) * MyAttack.lamada * MyAttack.K))
                else:
                    MyAttack.aff_update = v_s

                MyAttack.e = MyAttack.e + 1
                MyAttack.now_update = v_s
                
            if self.record_sign_match:
                if MyAttack.last_weight_vec is None:
                    updata_sign_match = np.mean(MyAttack.target_sign == np.sign(self.global_weights_vec))
                    MyAttack.updata_sign_match_list.append(updata_sign_match * 100)
                else:
                    updata_sign_match = np.mean(MyAttack.target_sign == np.sign(self.global_weights_vec - MyAttack.last_weight_vec))
                    MyAttack.updata_sign_match_list.append(updata_sign_match * 100)

                if self.global_epoch == self.args.epochs-1:
                    save_npy_date(self.args, MyAttack.updata_sign_match_list, self.args.npy_file_name, "updata_sign_match")

            # 保存一下信息给下一轮用
            MyAttack.last_weight_vec = self.global_weights_vec
            MyAttack.last_update = MyAttack.now_update
            return MyAttack.now_update
        else:
            return MyAttack.now_update

    @staticmethod
    def print_target_sign(target_sign):
        # 计算 -1 和 1 的数量
        num_neg_ones = np.sum(target_sign == -1)
        num_ones = np.sum(target_sign == 1)
        # 计算比例
        ratio_neg = num_neg_ones / (target_sign.size)
        ratio_pos = num_ones / (target_sign.size)

        print(f"Target Sign ratio_neg:{ratio_neg}|ratio_pos:{ratio_pos}|shape:{target_sign.shape}")

    @staticmethod
    def should_reject_H0(d, X, p=0.01):
        """
        判断是否拒绝零假设 H0（攻击失败），则不用调整参数

        参数:
        d (int): 全局模型的维度。
        p (float): 显著性水平。
        X (int): 符号匹配的维度数量。

        返回:
        bool: 如果 Pr(X >= X) <= p，则拒绝 H0，返回 True；否则返回 False。
        """
        # 计算 Pr(X >= X)
        prob_X_ge_X = 1 - binom.cdf(X - 1, d, 0.5)
        
        # 判断是否拒绝 H0
        if prob_X_ge_X <= p:
            return True
        else:
            return False
        