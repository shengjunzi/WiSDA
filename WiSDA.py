import paddle.nn as nn
from CSIResNet import ResNet ,BasicBlock
import SDALoss
import paddle
import math 
import pandas as pd




class WiSDA(nn.Layer):

    def __init__(self, num_classes=6, bottle_neck=False):
        super( WiSDA, self).__init__()
        self.feature_layers =ResNet(BasicBlock,[2,2,2,2])
        self.loss_function =  SDALoss.SDA_LOSS(class_num=num_classes)
        self.bottle_neck = bottle_neck
        if bottle_neck:
            self.bottle = nn.Linear(1024, 256)
            self.cls_fc = nn.Linear(256, num_classes)
        else:
            self.cls_fc = nn.Linear(512, num_classes)


    def forward(self, source, target, s_label, t_label,epoch):
        source = self.feature_layers(source)
        if self.bottle_neck:
            source = self.bottle(source)
        s_pred = self.cls_fc(source)
        target = self.feature_layers(target)
        t_pred = self.cls_fc(target)

        pseudo_label = nn.functional.softmax(t_pred, axis=-1)
        t_label1=paddle.argmax(pseudo_label, axis=-1)

        max_probs = paddle.max(pseudo_label, axis=-1)
        mean =paddle.mean(max_probs)
        var = paddle.var(max_probs)
        r = paddle.exp(-((max_probs-mean).pow(2))/(2*var))
        r = paddle.where(r > mean, paddle.ones_like(r), r)
        r1 = math.exp(-(epoch+5)/10)
        r1 = paddle.to_tensor(r1)

        #r=math.exp(-(epoch+10)/20)
        # if epoch ==2:
        #     print(r)
        # r = paddle.to_tensor(r)
       
        domain_loss = self.loss_function.sda_loss(source, target, s_label, t_label1,r,r1)
        if epoch ==1:
            df1 = pd.DataFrame(t_label1.numpy())
            df2 = pd.DataFrame(pseudo_label.numpy())
            df =pd.DataFrame(t_label.numpy())

            # 将DataFrame保存为Excel文件
            df.to_excel('t_label.xlsx', index=False)
            df1.to_excel('t_label1.xlsx', index=False)
            df2.to_excel('t_label2.xlsx', index=False)
       
        return s_pred,domain_loss

    def predict(self, x):
        x = self.feature_layers(x)
        if self.bottle_neck:
            x = self.bottle(x)
        return self.cls_fc(x)