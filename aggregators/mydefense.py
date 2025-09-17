from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from aggregators import aggregator_registry
from fl.models import get_model
from fl.models.model_utils import model2vec
from copy import deepcopy
import hdbscan
from plot_utils import *


@aggregator_registry
class MyDefense(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {}
        self.update_and_set_attr()

        self.s1 = np.zeros_like(model2vec(deepcopy(get_model(args))))
        self.s2 = np.zeros_like(self.s1)
        self.predict_weight_vec = None
        self.alpha = args.predict_alpha

        self.for_show_fscores = []
        self.num_clients = args.num_clients
        # rs分数都初始化为0
        self.rs_list = np.zeros(self.num_clients)
        
        # 奖惩分数
        self.reward = self.num_clients//2 - 1.0
        self.penalty = -(1.0 + self.num_clients//2)

        # 保存rs分数
        self.all_epoch = []
        self.all_rs = []

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs['last_global_model']
        _, now_gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model)
        
        self.now_global_weights_vec = kwargs['global_weights_vec']
        self.now_global_epoch = kwargs['global_epoch']
        self.last_aggregated_update = kwargs['aggregated_update']

        # 这里为了用前几轮全局模型初始化s，所以在前几轮是直接输出平均
        if self.now_global_epoch < self.args.predict_epochs:
            self.s1 += self.now_global_weights_vec * (1 / self.args.predict_epochs)
            self.s2 += self.now_global_weights_vec * (1 / self.args.predict_epochs)
            # 调整一下
            return wrapup_aggregated_grads(now_gradient_updates[self.args.num_adv:], self.args.algorithm, self.global_model, False)
        # 进入防御轮次
        else:
            # 预测下一轮的全局模型
            self.s1 = self.alpha * self.now_global_weights_vec + (1 - self.alpha) * self.s1
            self.s2 = self.alpha * self.s1 + (1 - self.alpha) * self.s2
            self.predict_weight_vec = (2 * self.s1 - self.s2 ) + (self.alpha / (1 - self.alpha)) * ( self.s1 - self.s2 )

            # 计算两个翻转方向
            old_direction = np.sign(self.last_aggregated_update)
            pre_direction = np.sign(self.predict_weight_vec - self.now_global_weights_vec)

            # 计算客户端的翻转分数
            all_fscores = []
            for update in now_gradient_updates:
                fscores = []
                old_flip = np.sign(np.sign(update) * (np.sign(update) - old_direction))
                fscores.append(np.sum(old_flip*(update**2)))
                pre_flip = np.sign(np.sign(update) * (np.sign(update) - pre_direction))
                fscores.append(np.sum(pre_flip*(update**2)))

                all_fscores.append(fscores)

            # 开始基于二维向量聚类
            cluster = hdbscan.HDBSCAN(metric=self.metric, algorithm="generic",
                                    min_cluster_size=self.args.num_clients//2+1, 
                                    min_samples=1, allow_single_cluster=True)
            
            cluster.fit(np.array(all_fscores, dtype=np.float64))
            all_set = set(list(range(self.args.num_clients)))

            ben_idx = [idx for idx, label in enumerate(cluster.labels_) if label == 0]
            ben_set = set(ben_idx)
            mal_set = all_set - ben_set
            mal_idx = list(mal_set)
            # print(f"MyDefense: mal clients: {mal_idx}")

            # 开始调整rs分数
            self.rs_list = self.rs_list * self.gamma
            self.rs_list[ben_idx] = self.rs_list[ben_idx] + self.reward
            self.rs_list[mal_idx] = self.rs_list[mal_idx] + self.penalty
            # print(f"rs_list: {self.rs_list}")

            # 是否记录rs分数
            if self.recored_rs:
                self.all_epoch.append(self.now_global_epoch)
                rs_mal_list = self.rs_list[:self.args.num_adv]
                rs_ben_list = self.rs_list[self.args.num_adv:]

                mal_mean = np.mean(rs_mal_list)
                mal_std = np.std(rs_mal_list)

                ben_mean = np.mean(rs_ben_list)
                ben_std = np.std(rs_ben_list)
                
                self.all_rs.append([[mal_mean, mal_std], [ben_mean, ben_std]])

            # rs分数用分段sorftmax归一化后作为权重
            if self.args.softmax_seg in "Min":
                c = np.min(self.rs_list)
                pos_data = self.rs_list[self.rs_list >= c]
                # T_pos = np.sqrt((np.max(pos_data) - np.min(pos_data)))+1 if len(pos_data) > 0 else 1.0
                # 用std做一个分段平滑
                T_pos = np.std(pos_data)+1 if len(pos_data) > 0 else 1.0
                T_neg = 1.0
                temperatures = np.where(self.rs_list > np.min(self.rs_list), T_pos, T_neg)
                scaled_rs = self.rs_list / temperatures
                exp_rs = np.exp(scaled_rs - np.max(scaled_rs))
                weights = exp_rs / np.sum(exp_rs)

            elif self.args.softmax_seg in "Max":
                c = np.max(self.rs_list)
                pos_data = self.rs_list[self.rs_list >= c]
                # T_pos = np.sqrt((np.max(pos_data) - np.min(pos_data)))+1 if len(pos_data) > 0 else 1.0
                # 用std做一个分段平滑
                T_pos = np.std(pos_data)+1 if len(pos_data) > 0 else 1.0
                T_neg = 1.0
                temperatures = np.where(self.rs_list > np.min(self.rs_list), T_pos, T_neg)
                scaled_rs = self.rs_list / temperatures
                exp_rs = np.exp(scaled_rs - np.max(scaled_rs))
                weights = exp_rs / np.sum(exp_rs)

            elif self.args.softmax_seg in "Normal":
                scaled_rs = self.rs_list
                exp_rs = np.exp(scaled_rs - np.max(scaled_rs))
                weights = exp_rs / np.sum(exp_rs)

            else:
                raise ValueError(f"Unknown softmax_seg: {self.args.softmax_seg}")

            # print(f"T_pos:{T_pos} | weights:{weights}")

            # 输出加权的更新
            assert weights.shape[0] == now_gradient_updates.shape[0], "shape Erron"
            agg_grad_updates = np.sum((weights[:, np.newaxis] * now_gradient_updates), axis=0)
            
            # 为了展示，只取第一个和最后一个客户端
            if self.show_3d:
                self.for_show_fscores.append([all_fscores[0], all_fscores[-1]])
            # 是否在最后一轮聚合后输出图片
            if self.show_3d and self.now_global_epoch == self.args.epochs-1:
                plot_fscores(self.args, self.for_show_fscores)

            if self.recored_rs and self.now_global_epoch == self.args.epochs-1:
                # print(f"{np.array(self.all_epoch).shape}")
                # print(f"{np.array(self.all_rs).shape}")
                save_npy(self.args, self.all_epoch, self.all_rs, self.args.npy_file_name, "rs-mal-and-ben")

            return wrapup_aggregated_grads(agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True)

def save_npy(args, all_epoch, all_rs, npy_file_name, npy_prefix):
    formatted_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    np.savez(f"/home/ymh/wxy/Sign-Fed/logs/data/{npy_file_name}/{npy_prefix}_{args.dataset}_{args.attack}_{args.defense}_{args.distribution}-{args.dirichlet_alpha}_adv-{args.num_adv/args.num_clients}_{formatted_time}.npy", array1=all_epoch, array2=all_rs)