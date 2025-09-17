import numpy as np
from global_utils import actor
from attackers.pbases.mpbase import MPBase
from attackers import attacker_registry
from fl.client import Client
from copy import deepcopy
from fl.models import get_model
from fl.models.model_utils import model2vec


@attacker_registry
@actor('attacker', 'model_poisoning', 'non_omniscient')
class MPAF(MPBase, Client):

    # 静态的随机初始化模型
    random_model_vec = None

    def __init__(self, args, worker_id, train_dataset, test_dataset):
        Client.__init__(self, args, worker_id, train_dataset, test_dataset)
        self.default_attack_params = {'lamda': 1e6}
        self.update_and_set_attr()

        if self.random_model_vec is None:
            MPAF.random_model_vec = model2vec(deepcopy(get_model(self.args)))

    def non_omniscient(self):
        update = float(self.lamda) * (MPAF.random_model_vec - self.global_weights_vec)
        return update
