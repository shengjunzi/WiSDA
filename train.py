import os
import random
from WiSDA import WiSDA

from LOAD import MyData
import paddle
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import paddle.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import math



#划分数据集
def ganlist(gaf_class):
    train_ratio=0.8
    train=open('data/data141031/CSIGAF'+str(gaf_class)+'/s_traindata.txt','w+')
    val=open('data/data141031/CSIGAF'+str(gaf_class)+'/s_validata.txt','w+')
    target=open('data/data141031/CSIGAF'+str(gaf_class)+'/target_data.txt','w+')
    source=open('data/data141031/CSIGAF'+str(gaf_class)+'/source_data.txt','w+')
    test=open('data/data141031/CSIGAF'+str(gaf_class)+'/test_data.txt','w+')
    root_dir='data/data141031/CSIGAF'+str(gaf_class)
    train_dir='data'
    path=os.path.join(root_dir,train_dir)
    img_path=os.listdir(path)
    for line in img_path:
        if line.split('-')[1]=='1':
            target.writelines(line)
            target.write('\r\n')
            # if random.uniform(0, 1) < train_ratio: 
            #     train.writelines(line)
            #     train.write('\r\n')   
            # else:
            #     val.writelines(line)
            #     val.write('\r\n')
        else:
            source.writelines(line)
            source.write('\r\n')
            if random.uniform(0, 1) < train_ratio-0.1: 
                train.writelines(line)
                train.write('\r\n')   
            else:
                if random.uniform(0, 1) < 0.5:
                    val.writelines(line)
                    val.write('\r\n')
                else:
                    test.writelines(line)
                    test.write('\r\n')      
    train.close()
    val.close()
    target.close()
    source.close()

def train_domain(target_loader,source_loader,batsize,EPOCH_NUM,class_criterion,model,lr1,lr2, model_name,save_name):
  
    scheduler = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr1, milestones=[4, 8,12,16], gamma=0.1)
    scheduler1 = paddle.optimizer.lr.MultiStepDecay(learning_rate=lr2, milestones=[4, 8, 12, 16], gamma=0.1)
    optimizer_F=paddle.optimizer.Adam(parameters=model.feature_layers.parameters(),learning_rate=scheduler)
    optimizer_D=paddle.optimizer.Adam(parameters=model.cls_fc.parameters(),learning_rate=scheduler1)

    best_acc = 0
    vali_acc = []
    train_acc =[]
    train_loss=[]
    vali_loss=[]
    use_gpu = True
    paddle.device.set_device('gpu:0') if use_gpu else paddle.device.set_device('cpu')
    print("start traing")
    for epoch in range(EPOCH_NUM):
        model.train()
        t_model=WiSDA(6)
        train_loss_b=[]
        vali_loss_b=[]
        train_acc_b=[]
        vali_acc_b=[]
        tvali_acc=[]
        correct_t = 0
        domain_loss_b=[]
        loss_b=[]
        t_loss=[]
        for batch_id, (data,t_data) in enumerate(zip(source_loader(),target_loader())):
            s_img = data[0]
            
            s_label = data[1]
            t_label=t_data[1]
            t_img=t_data[0]
           
         
            # 1 : 训练
            s_feature,domain_loss =model(s_img,t_img,s_label,t_label,epoch)
        
         

            s_classloss= class_criterion(s_feature, s_label) #源域损失
           
            correct_t=paddle.metric.accuracy(s_feature, s_label) #源域准确率
    
            train_acc_b.append(correct_t.numpy())
            train_loss_b.append(s_classloss)
       
            domain_loss_b.append(domain_loss)
           

       
            lambd = 2/(1 + math.exp(-10 * (epoch+2) / 20)) - 1
            loss=s_classloss+lambd*domain_loss
        
        
            #反向传播
            loss.backward()
            #跟新参数
            optimizer_F.step()
            optimizer_D.step()
            #梯度清零
            optimizer_D.clear_grad()
            optimizer_F.clear_grad()
        scheduler.step()
        scheduler1.step()
        train_loss.append(np.mean(train_loss_b))
        train_acc.append(np.mean(train_acc_b))
        model.eval()
        for batch_id, data in enumerate(target_loader):
            img = data[0]
            label = data[1] 
        # 计算模型输出
            predict_label =model.predict(img)
        # 计算损失函数
            loss = class_criterion(predict_label, label)
            vali_loss_b.append(loss.numpy())
            correct_t=paddle.metric.accuracy(predict_label,label)
            vali_acc_b.append(correct_t.numpy())
        print("epoch: {},  trainloss is: {},vali_loss is:{}，domainloss is: {}".format(epoch,  np.mean(train_loss_b), np.mean(vali_loss_b),np.mean(domain_loss_b)))
        print("epoch: {},  trainacc is: {},valiacc is: {}".format(epoch,  np.mean(train_acc_b),np.mean(vali_acc_b)))
        vali_loss.append( np.mean(vali_loss_b))
        vali_acc.append(np.mean(vali_acc_b))
        if np.mean(vali_acc_b) > best_acc:
            best_acc = np.mean(vali_acc_b)
            best_epoch = epoch
            paddle.save(model.state_dict(), "domain_model/model_"+model_name+".pdparams")
            print("更新模型")
    train_loss = pd.DataFrame(train_loss)
    train_acc = pd.DataFrame(train_acc)
    domain_loss1 = pd.DataFrame(domain_loss1)
    vali_loss = pd.DataFrame(vali_loss)
    vali_acc = pd.DataFrame(vali_acc)
    train_loss.to_excel(save_name+"/train_loss4.xlsx")
    train_acc.to_excel(save_name+"/train_acc4.xlsx")
    #domain_loss.to_excel(save_name+"/domian_loss3.xlsx")
    vali_loss.to_excel(save_name+"/vali_loss4.xlsx")
    vali_acc.to_excel(save_name+"/vali_acc4.xlsx")



  
if __name__ == '__main__':
    data_num=3
    ganlist(data_num)
    root_dir='data/data141031/CSIGAF3/data'
    train_dir='s_traindata'
    vali_dir='s_validata'
    tar_dir='target_data'
    test_dir='test_data'
    source_dir='source_data'

    testdata=MyData(root_dir,test_dir,data_num)
    sourcedata=MyData(root_dir,source_dir,data_num)
    targetdata=MyData(root_dir,tar_dir,data_num)
    traindata=MyData(root_dir,train_dir,data_num)
    validata=MyData(root_dir,vali_dir,data_num)
 
    
    print("源域数据量:{},目标域数据量:{}".format(len(sourcedata),len(targetdata)))
    print("训练数据量:{},验证域数据量:{}，测试数据量:{}".format(len(traindata),len(validata),len(testdata)))
    batsize=128
    EPOCH_NUM=15
    class_criterion = nn.CrossEntropyLoss()
    target_loader=paddle.io.DataLoader(targetdata,batch_size=batsize,shuffle=True,drop_last=True,num_workers = 2)
    source_loader=paddle.io.DataLoader(sourcedata,batch_size=batsize,shuffle=True,drop_last=True,num_workers = 2)
   

    # train_loader=paddle.io.DataLoader(traindata,batch_size=batsize,shuffle=True,drop_last=True)
    # vali_loader=paddle.io.DataLoader(validata,batch_size=batsize,shuffle=True,drop_last=True)
    # test_loader = paddle.io.DataLoader(testdata, batch_size=batsize, shuffle=False,drop_last=True)


    model =DSAN(6)
    lr1 = 0.001
    lr2 = 0.01
    model_name="domain3f"
    save_name = "result/lc"
    train_domain(target_loader,source_loader,batsize,EPOCH_NUM,class_criterion,model,lr1,lr2,model_name,save_name)
