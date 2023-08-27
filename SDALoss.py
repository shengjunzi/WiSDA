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
        x = paddle.ones(shape=[m])
        #print(x)
        rx =  paddle.concat([x, r], axis=0)
        rx1 = paddle.unsqueeze(rx, axis=1)
        rx2 = paddle.unsqueeze(rx, axis=0)

        #r = paddle.expand(r, shape=[2*m, m])
        rx3 = paddle.matmul(rx1,rx2)
        #print(rx3)
        sim = sim*rx3
        #sim[:,m:2*m] = sim[:,m:2*m]*r
        #print(sim)
        #print(r)b
        
        #print(sim)
       
      

        
        t_label = paddle.reshape(t_label,[m,1])

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
        #print(matrix)

        nominator = paddle.exp(sim/r1)
        dnominator = paddle.exp(sim/r1)




        nominator = nominator*mask_p
        # print(dnominator)
        # print(dnominator*mask_n)
        # print(dnominator*mask_n*matrix)

        dnominator =dnominator*mask_n

        a=paddle.mean(nominator, axis=1)
        b=paddle.sum(dnominator, axis=1)
        loss_partial = -paddle.log(a/b) 

        # loss_partial =  paddle.unsqueeze(loss_partial, axis=0)
        # loss_partial[:,m:2*m] = loss_partial[:,m:2*m]*r
        loss = paddle.mean(loss_partial) 

        return loss









       



