from torchsummary import summary
from fl.models.resnet import resnet10
from fl.models.resnet import resnet18
from fl.models.resnet import resnet34
from fl.models.lenet5 import lenet
from fl.models.mycnn import MyCnn


model = MyCnn()
model.cuda()
summary(model, input_size=(3, 32, 32))