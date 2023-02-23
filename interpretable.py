
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision
import random
import numpy as np
from utils import progress_bar
import copy 
import torch.optim as optim
from torchvision import datasets, transforms
import os
import sys 
import time
import Helper
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    InputXGradient,
    Saliency,
    NoiseTunnel
)
from captum.attr import visualization as viz

import matplotlib.pyplot as plt
import numpy as np

use_cuda =  torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def train(args,epoch,model,trainloader,optimizer,criterion,criterionKDL,k, Name=None):
    print(model)
    if(Name==None):
        print('\nEpoch: %d' % epoch)
    else:
        print('\nEpoch: %d' % epoch, Name)


    if epoch <= 1:
      k_min = 0
      k_max = 0
      m= 0
    elif 1 < epoch <= 2:
      k_min = 100
      k_max = 450
      m= 5
    elif 2 < epoch:
      k_min = 150 
      k_max = 900
      m= 5

    train_loss = 0
    Kl_loss=0
    Model_loss=0
    correct = 0
    correctAugmented=0
    total = 0
    model.train()
    softmax = nn.Softmax(dim=1)
    if(args.RandomMasking):
        maskType="randomMask"
    else:
        maskType="meanMask"

    number_of_estimate=  10 
    counter_True = np.zeros((number_of_estimate,10))
    counter_All = np.zeros((number_of_estimate,10))
    counter_prcentage= np.zeros((number_of_estimate,10))
    prediction =np.zeros(number_of_estimate)
    #target =np.zeros(number_of_estimate)
  
    all_grads = []

    for batch_idx, (data, target) in enumerate(trainloader):
        data, target = data.to(device), target.to(device)
        data.requires_grad = True
        k_batch = k[batch_idx]
        #print(k_batch.shape)
        #print("batch-size",batch_size)
        if(args.featuresDroped!=0):# and epoch > 10:
            model.eval()
            numberOfFeatures = int(args.featuresDroped*data.shape[1]*data.shape[2]*data.shape[3])

            tempData=data.clone()
            #print("shape", tempData.shape)
            saliency = Saliency(model)
            grads= saliency.attribute(data, target, abs=False).mean(1).detach().cpu().to(dtype=torch.float)
            #print(grads.shape)
            grads_=grads[7].reshape(1024)
            range_grads = max(grads_) - min(grads_)
            interval_width = range_grads / 10
            intervals = np.linspace(min(grads_), max(grads_), 11)
            samples_in_intervals, _ = np.histogram(grads_, bins=intervals)
            #print(samples_in_intervals)



            #print(grads[0].shape)
            torch.save(grads[0], "/maskedImage"+ str(epoch)+'_'+str(batch_idx))
            all_grads.append(grads.cpu().detach().numpy())

            #output = model(grads[0])
            if batch_idx == 0:
              torch.save(grads[7],'grads.pt')
            
            #print(counter_True.shape)
            #if epoch == 10 and batch_idx == 0:
            #  for k in range(len(samples_in_intervals)):
            if(args.abs):
                grads=grads.abs()
            if(args.isMNIST):
                torch.save(tempData[7],'temp_Data_sample.pt')
                temp_Data=tempData.view(tempData.shape[0], -1).detach()
                tempGrads= grads.view(grads.shape[0], -1)
                #numberOfFeatures = samples_in_intervals[k] 
                #print("numberOfFeatures",numberOfFeatures)

                #for idx in range(temp_Data.shape[0]):
                #   if(args.RandomMasking):
                #     output = model(input_eval[idx].unsqueeze(0))

                #print("tempGrads.shape",tempGrads.shape)
                #values,indx = torch.topk(tempGrads, numberOfFeatures, dim=1,largest=False)

                for idx in range(temp_Data.shape[0]):
                    #print(k_batch[idx])
                    values,indx = torch.topk(tempGrads[idx], k_batch[idx],largest=False)
                    #print(indx)

                    if(args.RandomMasking):
                        min_=torch.min(temp_Data[idx]).item()
                        max_=torch.max(temp_Data[idx]).item()


                        randomMask = np.random.uniform(low=min_, high=max_, size=(len(indx),))
                        #print("1")
                        temp_Data[idx][indx]= torch.Tensor(randomMask).to(device)
                        #print(temp_Data[idx].shape)
                        input_eval=temp_Data.view(data.shape).detach()
                        #print(input_eval[idx].shape)
                        #print("data",data[idx].shape)
                        #print("numberOfFeatures",numberOfFeatures)               
                        output = model(input_eval[idx].unsqueeze(0))
                        #print("output",output)
                        pred_out, pred_index = torch.max(output, 1)
                        #print(pred_index)
                        #print("target shape",target.shape)
                        #print("target index",target[idx])
                        #counter_All[k][target[idx]]+=1
                        #if pred_index==target[idx]:
                        #  counter_True[k][target[idx]]+=1
                        
                        if idx == 7 or idx == 1 or idx == 2 or idx == 4 or idx == 10 or idx == 20 or idx == 30 or idx == 40:
                          #prediction[k]=pred_index
                          #target= target[idx]
                          #torch.save(temp_Data[idx],'temp_Data_sample.pt')
                          #torch.save(grads[7],'grads.pt')
                          torch.save(grads[idx],'grads'+str(idx))#+str(k))
                          torch.save(output,'output'+str(idx))#+str(k))
                          torch.save(target[idx],'target'+str(idx))#+str(k))
                          #torch.save(prediction,'temp_Data_Predict.pt')
                          #torch.save(prediction,'temp_Data_Predict.pt')
                          #torch.save(target[idx],'temp_Data_Label.pt')
                        

                    else:
                        temp_Data[idx][indx[idx]]= data[0,0,0,0]
                        print(temp_Data[idx].shape)
                        input_eval=temp_Data.view(data.shape).detach()
                        print(input_eval[idx].shape)
                        output = model(input_eval[idx].unsqueeze(0))
                        print("output",output)
                        pred_out, pred_index = torch.max(output, 1)
                        print(pred_out, pred_index)
                        print("target shape",target[indx])
                        print(pred_index==target[idx])
            else:
                for idx in range(tempData.shape[0]):
                    singleMask=  Helper.get_SingleMask(args.featuresDroped,grads[idx], remove_important=False)
                    tempData[idx] = Helper.fill_SingleMask(tempData[idx],singleMask,maskType)


            maskedInputs=tempData.view(data.shape).detach()
            model.train()
        

       
        optimizer.zero_grad()
        output= model(data)
        #print("output.shape",output.shape)
        #print("output[0]",output[0])
        
        SoftmaxOutput=softmax(output)
        #print("SoftmaxOutput.shape",SoftmaxOutput.shape)
        #print("SoftmaxOutput[0]",SoftmaxOutput[0])

        Modelloss = criterion(output, target)
        loss=Modelloss
        Model_loss+=Modelloss.item()

        
        if(args.featuresDroped!=0):
            target_one_hot= torch.nn.functional.one_hot(target,10)
            #print(target_one_hot.shape)
            maskedOutputs= model(maskedInputs)
            #print("maskedOutputs.shape",maskedOutputs.shape)
            #maskedOutputs2 = F.log_softmax(maskedOutputs, dim=1)
            #print(maskedOutputs2)
            #print("update k")
            target_KL = target_one_hot.float() / target_one_hot.sum().float()
            masked_kl = maskedOutputs.float()
            output_kl = output.float()

            for i in range (500):
              l = torch.sign(criterionKDL(masked_kl[i],target_KL[i]) - criterionKDL(output[i],target_KL[i])) * (criterionKDL(masked_kl[i],target_KL[i]) -criterionKDL(output[i],target_KL[i]))
              #print("kl1",criterionKDL(masked_kl[i],target_KL[i]))
              #print("kl2", criterionKDL(output[i],target_KL[i]))
              #print(l)
              #j = max(k_min, min(k_max, m * l))
              k_batch[i] =  max(k_min, min(k_max, k_batch[i] + int(m * l)))
              #print(k_batch[i])


            KDL_m= criterionKDL(masked_kl,target_KL)
            #print("KDL_m",KDL_m)

            KDL_o= criterionKDL(output,target_KL)
            #print("KDL_o",KDL_o)

            k[batch_idx] = k_batch 

            if(args.isMNIST):
                KLloss = criterionKDL(maskedOutputs,output)

            else:
                maskedOutputs = F.log_softmax(maskedOutputs, dim=1)
                SoftmaxOutput=softmax(output)
                KLloss = criterionKDL(maskedOutputs,SoftmaxOutput)
           
            Kl_loss+=KLloss.item()
            loss=loss + KLloss
        

        #else:
            #KLloss=0
        #KLloss=0

        loss.backward()
        optimizer.step()


        train_loss += loss.item()
        predicted = output.argmax(dim=1, keepdim=True) 
        total += target.size(0)
        correct += predicted.eq(target.view_as(predicted)).sum().item()




        if args.featuresDroped!=0 :
        
            progress_bar(batch_idx, len(trainloader), '# %.1f Loss: %.3f |  Modelloss %.3f  KLloss %.3f | Acc: %.3f'
                 % (args.featuresDroped,train_loss/(batch_idx+1),Model_loss/(batch_idx+1), Kl_loss /(batch_idx+1), 100.*correct/total))
        else:
            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total ))

    """
    print("counter_All",counter_All)
    print("counter_True",counter_True)
    #print("counter_prcentage",counter_prcentage)
    if epoch == 10:
      for i in range(len(counter_All)):
        for j in range(len(counter_All[i])):
          counter_prcentage[i][j]=counter_True[i][j]/counter_All[i][j]
    print("counter_prcentage",counter_prcentage)
    torch.save(counter_prcentage,'counter_prcentage.pt')
    """
    print("k",np.min(k),np.mean(k),np.median(k),np.max(k))
    torch.save(k,'k'+str(epoch))
    return model , Model_loss , Kl_loss , np.array(all_grads, dtype= float), k



def test(args,epoch,model,testloader,criterion,criterionKDL,best_acc,best_epoch,returnMaskedAcc=False):


    softmax = nn.Softmax(dim=1)

    model.eval()
    test_loss = 0
    Kl_loss=0
    correct = 0
    total = 0
    Model_loss=0
    correctAugmented=0
    augmentedAcc=0

    if(args.RandomMasking):
        maskType="randomMask"
    else:
        maskType="meanMask"

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(testloader):
            data, target = data.to(device), target.to(device)
            data.requires_grad = True


            if(args.featuresDroped!=0):
                numberOfFeatures = int(args.featuresDroped*data.shape[1]*data.shape[2]*data.shape[3])

                tempData=data.clone()
                saliency = Saliency(model)
                grads= saliency.attribute(data, target, abs=False).mean(1).detach().cpu().to(dtype=torch.float)
                if(args.abs):
                    grads=grads.abs()
                if(args.isMNIST):
                    tempData=tempData.view(tempData.shape[0], -1).detach()
                    tempGrads= grads.view(grads.shape[0], -1)
                    values,indx = torch.topk(tempGrads, numberOfFeatures, dim=1,largest=False)
                


                    for idx in range(tempData.shape[0]):
                        if args.RandomMasking:
                            min_=torch.min(tempData[idx]).item()
                            max_=torch.max(tempData[idx]).item()
                            randomMask = np.random.uniform(low=min_, high=max_, size=(len(indx[idx]),))
                            tempData[idx][indx[idx]]= torch.Tensor(randomMask).to(device)
                        else:
                            tempData[idx][indx[idx]]= data[0,0,0,0]
                else:
                             


                    for idx in range(tempData.shape[0]):
                        singleMask=  Helper.get_SingleMask(args.featuresDroped,grads[idx], remove_important=False)

                        tempData[idx] = Helper.fill_SingleMask(tempData[idx],singleMask,maskType)


                maskedInputs=tempData.view(data.shape).detach()
        

                maskedOutputs= model(maskedInputs)


            outputs= model(data)

            Modelloss = criterion(outputs, target)
            Model_loss+=Modelloss.item()
            test_loss += Modelloss.item()

            if(args.featuresDroped!=0):

                if(args.isMNIST):
                    KLloss = criterionKDL(maskedOutputs,outputs)
                else:
    
                    maskedOutputs = F.log_softmax(maskedOutputs, dim=1)
                    SoftmaxOutput=softmax(outputs)
                    KLloss = criterionKDL(maskedOutputs,SoftmaxOutput)



                Kl_loss+=KLloss.item()
                test_loss+=KLloss.item()



            predicted = outputs.argmax(dim=1, keepdim=True) 
            total += target.size(0)
            correct += predicted.eq(target.view_as(predicted)).sum().item()




            if(args.featuresDroped!=0):
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f | best: Acc %.3f epoch %d'
                     % (test_loss/(batch_idx+1), 100.*correct/total,best_acc,best_epoch))
            else:
                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f  | best: Acc %.3f epoch %d '
                         % (test_loss/(batch_idx+1), 100.*correct/total,best_acc,best_epoch))

    # Save checkpoint.
    acc = 100.*correct/total
    Kl_loss=Kl_loss/(batch_idx+1)
    
    if(returnMaskedAcc):
        return acc, augmentedAcc, Kl_loss
    else:
        return acc ,Kl_loss