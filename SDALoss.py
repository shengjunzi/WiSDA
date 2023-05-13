import paddle.nn.functional as F
from sklearn.cluster import DBSCAN
import paddle
import paddle.nn as nn
import numpy as np
import math


class SDA_LOSS(nn.Layer):
    def __init__(self,class_num = 6):
        super(SDA_LOSS, self).__init__()
        self.class_num = class_num
   

    def sda_loss(self,source, target,s_label, t_label,r,r1):
        m, n = source.shape
        _, p = target.shape
        source /= paddle.norm(source, axis=1).unsqueeze(1)
        target /= paddle.norm(target, axis=1).unsqueeze(1)

        result_horizontal = paddle.concat([source, target], axis=0)
        #print(result_horizontal )
        sim = paddle.matmul(result_horizontal,result_horizontal.T)
        #print(sim)
        #r = paddle.unsqueeze(r, axis=0)
        #print(r)
        r = paddle.expand(r, shape=[2*m, m])

        
        #print(sim)
        #print(r)
        sim[:,128:256] = sim[:,128:256]*r
        #print(sim)
      

        
        t_label = paddle.reshape(t_label,[128,1])

        label = paddle.concat([s_label, t_label], axis=0)
        label = label.squeeze()
        mask_p = (label.unsqueeze(-1) == label.unsqueeze(0)).astype('bool')
        mask_n = (label.unsqueeze(-1) != label.unsqueeze(0)).astype('bool')
        #print(label)
        #print(mask_p)

        matrix = np.ones((2*m, 2*m), dtype=bool)
        # 将矩阵对角线上的元素设置为False
        np.fill_diagonal(matrix, False)
        #print(matrix)
        matrix = paddle.to_tensor(matrix)

        nominator = paddle.exp(sim*mask_p/r1)
        dnominator = paddle.exp(sim*mask_n*matrix/r1)

        a=paddle.mean(nominator, axis=1)
        b=paddle.sum(dnominator, axis=1)
        loss_partial = -paddle.log(a/b) 
        loss = paddle.sum(loss_partial) / (2*m)

        return loss
