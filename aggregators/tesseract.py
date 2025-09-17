from aggregators.aggregatorbase import AggregatorBase
import numpy as np
from aggregators.aggregator_utils import prepare_updates, wrapup_aggregated_grads
from aggregators import aggregator_registry


@aggregator_registry
class TESSERACT(AggregatorBase):
    def __init__(self, args, **kwargs):
        super().__init__(args)
        self.default_defense_params = {}
        self.update_and_set_attr()

        # rs分数都初始化为0
        self.rs_list = np.zeros(args.num_clients)

        self.adv_rate = 0.1
        
        # 奖惩分数
        self.penalty = 1.0 - 2.0 * (((self.adv_rate)*args.num_clients)/args.num_clients)
        self.reward = 1.0 - self.penalty

    def aggregate(self, updates, **kwargs):
        self.global_model = kwargs['last_global_model']
        _, now_gradient_updates = prepare_updates(
            self.args.algorithm, updates, self.global_model)
        
        self.now_global_weights_vec = kwargs['global_weights_vec']
        self.now_global_epoch = kwargs['global_epoch']
        self.last_aggregated_update = kwargs['aggregated_update']

        # 计算翻转方向
        old_direction = np.sign(self.last_aggregated_update)

        # 计算客户端的翻转分数
        flip_local = []
        for update in now_gradient_updates:
            old_flip = np.sign(np.sign(update) * (np.sign(update) - old_direction))
            flip_local.append(np.sum(old_flip*(update**2)))

        argsorted = np.argsort(flip_local)
        cmax = int((self.adv_rate)*self.args.num_clients)
        if cmax > 0:
            self.rs_list[argsorted[cmax:-cmax]] = self.rs_list[argsorted[cmax:-cmax]] + self.reward
            self.rs_list[argsorted[:cmax]] = self.rs_list[argsorted[:cmax]] - self.penalty
            self.rs_list[argsorted[-cmax:]] = self.rs_list[argsorted[-cmax:]] - self.penalty  

        # print(f"rs_list: {self.rs_list}")

        # rs分数用分段sorftmax归一化后作为权重
        exp_rs = np.exp(self.rs_list)
        weights = exp_rs / np.sum(exp_rs)

        # 输出加权的更新
        assert weights.shape[0] == now_gradient_updates.shape[0], "shape Erron"
        agg_grad_updates = np.sum((weights[:, np.newaxis] * now_gradient_updates), axis=0)
        
        return wrapup_aggregated_grads(agg_grad_updates, self.args.algorithm, self.global_model, aggregated=True)