import os
import sys

sys.path.append("..")
import numpy as np
# from generate import generate_vmatrix
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.loss import NTXentLoss, SupConLoss

ft_perc = '50p'

class FeatureKLDivergenceLoss(nn.Module):
    def forward(self, input1, input2, input3):
        # 计算KL散度
        print(input1.shape)
        print(input2.shape)
        print(input3.shape)

        zero_column = torch.zeros((128, 1))

        # 使用torch.cat()将零列添加到矩阵中，dim=1表示在列维度上拼接
        input2 = torch.cat((input2, zero_column), dim=1)
        input3 = torch.cat((input3, zero_column), dim=1)


        kl_divergence_1 = torch.sum(input2 * (torch.log(input2) - torch.log(input1)))
        kl_divergence_2 = torch.sum(input3 * (torch.log(input3) - torch.log(input1)))
        kl_divergence_3 = torch.sum(input3 * (torch.log(input3) - torch.log(input2)))

        # 计算三个输入两两之间的KL散度之和
        kl_divergence_sum = -(kl_divergence_1 + kl_divergence_2 + kl_divergence_3)

        return kl_divergence_sum

def Trainer(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")
    if (training_mode == f"ft_{ft_perc}_and_generateV") or (training_mode == "self_supervised_decoder"):
        return
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _, _, _, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device, training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _, _, _, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    logger.debug("\n################## Training is Done! #########################")


def Trainer_ft(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    logger.debug("Training started ....")

    Z12 = []
    Z13 = []
    Z23 = []

    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        loss_task_1, loss_task_2, loss_task_3, train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)
        valid_loss, valid_acc, _, _, _, _, _, _ = model_evaluate(model, temporal_contr_model, valid_dl, device,
                                                                 training_mode)
        if (training_mode != "self_supervised") and (training_mode != "SupCon"):
            scheduler.step(valid_loss)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train Loss     : {train_loss:2.4f}\t | \tTrain Accuracy     : {train_acc:2.4f}\n'
                     f'Valid Loss     : {valid_loss:2.4f}\t | \tValid Accuracy     : {valid_acc:2.4f}')

        if epoch % 5 == 0:
            torch.save(model.state_dict(), os.path.join(experiment_log_dir, 'model_epoch{}.pth'.format(epoch + 1)))
            for i in range(2):
                model_train_generateZ(model, temporal_contr_model,model_optimizer, temp_cont_optimizer,criterion, train_dl,
                                                                              config, device,
                                                                              training_mode,i+1)

                _, _, loss_task1_next, loss_task2_next, loss_task3_next = model_evaluate_generateZ(model, temporal_contr_model, valid_dl,
                                                                             device,
                                                                             training_mode)

                if i == 0:
                    z12 = 1 - (loss_task_2 / loss_task2_next)
                    z13 = 1 - (loss_task_3 / loss_task3_next)
                    Z12.append(z12)
                    Z13.append(z13)

                if i == 1:
                    z23 = 1 - (loss_task_3 / loss_task3_next)
                    Z23.append(z23)

            checkpoint = torch.load(os.path.join(experiment_log_dir, 'model_epoch{}.pth'.format(epoch + 1)))
            model.load_state_dict(checkpoint)


    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    if (training_mode != "self_supervised") and (training_mode != "SupCon"):
        # evaluate on the test set
        logger.debug('\nEvaluate on the Test set:')
        test_loss, test_acc, _, _, _, _, _, _ = model_evaluate(model, temporal_contr_model, test_dl, device, training_mode)
        logger.debug(f'Test loss      :{test_loss:2.4f}\t | Test Accuracy      : {test_acc:2.4f}')

    total = 0
    count = 0
    for number in Z12:
        total += number
        count += 1
    average_12 = total / count

    total = 0
    count = 0
    for number in Z13:
        total += number
        count += 1
    average_13 = total / count

    total = 0
    count = 0
    for number in Z23:
        total += number
        count += 1
    average_23 = total / count
    print(average_12,average_13,average_23)
    logger.debug("\n################## Training is Done! #########################")
    return average_12, average_13, average_23


def Trainer_self_supervised_decoder(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    logger.debug("Training started ....")
    criterion = FeatureKLDivergenceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_loss, train_acc = model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer,
                                            criterion, train_dl, config, device, training_mode)

    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'model_state_dict': model.state_dict(),
                'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    logger.debug("\n################## Training is Done! #########################")


def model_train(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, label1, label2, label3, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, label1, label2, label3 = data.float().to(device), label1.long().to(device), label2.long().to(device), label3.long().to(device)
        aug1, aug2 = aug1.float().to(device), aug2.float().to(device)
        print(data.shape)
        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        if training_mode == "self_supervised_decoder":
            output = model(data)
            prediction_task_1, prediction_task_2, prediction_task_3, features = output
            loss = criterion(prediction_task_1, prediction_task_2, prediction_task_3)

        if training_mode == f"ft_{ft_perc}_and_generateV":
            output = model(data)
            prediction_task_1, prediction_task_2, prediction_task_3, features = output
            # loss = criterion(prediction_task_2, label2)
            loss_task1 = criterion(prediction_task_1, label1)
            loss_task2 = criterion(prediction_task_2, label2)
            loss_task3 = criterion(prediction_task_3, label3)
            loss = (loss_task1 + loss_task2 + loss_task3) / 3
            total_acc.append((label1.eq(prediction_task_1.detach().argmax(dim=1)) * 0.333 + label2.eq(
                prediction_task_2.detach().argmax(dim=1)) * 0.333 + label3.eq(
                prediction_task_3.detach().argmax(dim=1)) * 0.333).float().mean())

        if training_mode == "self_supervised" or training_mode == "SupCon":
            '''
            predictions 的形状应该是 (batch_size, num_classes),表示对batch中每一个样本的预测情况
            features1的形状应该是 (batch_size, final_out_channels, features_len)，其中 final_out_channels 是最后一个卷积块的输出通道数，features_len 是特征长度。
            '''
            predictions1_1, predictions1_2, predictions1_3, features1 = model(aug1) #
            predictions2_1, predictions2_2, predictions2_3, features2 = model(aug2)

            # normalize projection feature vectors  归一化
            features1 = F.normalize(features1, dim=1)
            features2 = F.normalize(features2, dim=1)

            '''
            计算 NCE 损失，并将计算得到的损失值和经过投影头处理后的结果作为输出返回
            temp_cont_loss是一个值，代表两个向量之间的距离
            temp_cont_feat是将预测的c_t经过一个维度变换并且可学习的投影头映射到configs.final_out_channels // 4 维度
            此处的loss1和loss2是预测的loss，原文中的L_TC
            '''

            temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(features1, features2)
            temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(features2, features1)


            if training_mode == "self_supervised":
                lambda1 = 1
                lambda2 = 0.7
                '''
                输入两个维度为(batch_size, representation_dim)的数据，输出其loss值
                此处loss是两个预测之间的距离，原文中的L_CC
                '''
                nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                               config.Context_Cont.use_cosine_similarity)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                       nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2


            elif training_mode == "SupCon":
                lambda1 = 0.01
                lambda2 = 0.1
                Sup_contrastive_criterion = SupConLoss(device)

                supCon_features = torch.cat([temp_cont_feat1.unsqueeze(1), temp_cont_feat2.unsqueeze(1)], dim=1)
                loss = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + Sup_contrastive_criterion(supCon_features,
                                                                                                 label1, label2, label3) * lambda2
        else:
            '''
            在其余模式下output仅为encoder层的输出
            '''
            output = model(data)
            prediction_task_1, prediction_task_2, prediction_task_3, features = output
            # loss = criterion(prediction_task_2, label2)
            loss = (criterion(prediction_task_1, label1) + criterion(prediction_task_2, label2) + criterion(prediction_task_3, label3))
            total_acc.append((label1.eq(prediction_task_1.detach().argmax(dim=1)) * 0.333 + label2.eq(prediction_task_2.detach().argmax(dim=1)) * 0.333 + label3.eq(prediction_task_3.detach().argmax(dim=1)) * 0.333).float().mean())

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        temp_cont_optimizer.step()

    total_loss = torch.tensor(total_loss).mean()

    if (training_mode == "self_supervised") or (training_mode == "SupCon"):
        total_acc = 0
        return total_loss, total_acc

    if (training_mode == f"ft_{ft_perc}_and_generateV"):
        total_acc = torch.tensor(total_acc).mean()
        return loss_task1, loss_task2, loss_task3, total_loss, total_acc
    else:
        total_acc = torch.tensor(total_acc).mean()
        return total_loss, total_acc

def model_train_generateZ(model, temporal_contr_model, model_optimizer, temp_cont_optimizer, criterion, train_loader, config,
                device, training_mode, i):
    total_loss = []
    total_acc = []
    model.train()
    temporal_contr_model.train()

    for batch_idx, (data, label1, label2, label3, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, label1, label2, label3 = data.float().to(device), label1.long().to(device), label2.long().to(device), label3.long().to(device)

        # optimizer
        model_optimizer.zero_grad()
        temp_cont_optimizer.zero_grad()

        output = model(data)
        prediction_task_1, prediction_task_2, prediction_task_3, features = output
        # loss = criterion(prediction_task_2, label2)
        loss_task1 = criterion(prediction_task_1, label1)
        loss_task2 = criterion(prediction_task_2, label2)
        loss_task3 = criterion(prediction_task_3, label3)
        loss = (loss_task1 + loss_task2 + loss_task3) / 3
        total_acc.append((label1.eq(prediction_task_1.detach().argmax(dim=1)) * 0.333 + label2.eq(
                prediction_task_2.detach().argmax(dim=1)) * 0.333 + label3.eq(
                prediction_task_3.detach().argmax(dim=1)) * 0.333).float().mean())

        total_loss.append(loss.item())

        if i ==1:
            loss_task1.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()

        if i ==2:
            loss_task2.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()

        if i ==3:
            loss_task3.backward()
            model_optimizer.step()
            temp_cont_optimizer.step()




def model_evaluate(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()
    out_task_1 = np.array([])
    out_task_2 = np.array([])
    out_task_3 = np.array([])
    trg_task_1 = np.array([])
    trg_task_2 = np.array([])
    trg_task_3 = np.array([])

    with torch.no_grad():
        for data, label1, label2, label3, _, _ in test_dl:
            data, label1, label2, label3 = data.float().to(device), label1.float().long().to(device), label2.float().long().to(device), label3.float().long().to(device)
            # print(label1.shape)
            # print(label2.shape)

            if (training_mode == "self_supervised") or (training_mode == "SupCon") or (training_mode == "self_supervised_decoder"):
                pass

            else:
                output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                prediction_task_1, prediction_task_2, prediction_task_3, features = output
                # print(prediction_task_1.shape)
                loss_task1 = criterion(prediction_task_1, label1)
                loss_task2 = criterion(prediction_task_2, label2)
                loss_task3 = criterion(prediction_task_3, label3)

                loss = loss_task1 + loss_task2 + loss_task3
                total_acc.append((label1.eq(prediction_task_1.detach().argmax(dim=1)) * 0.333 + label2.eq(
                    prediction_task_2.detach().argmax(dim=1)) * 0.333 + label3.eq(prediction_task_3.detach().argmax(dim=1)) * 0.333).float().mean())
                total_loss.append(loss.item())

                pred_1 = prediction_task_1.max(1, keepdim=True)[1]  # get the index of the max log-probability
                pred_2 = prediction_task_2.max(1, keepdim=True)[1]
                pred_3 = prediction_task_3.max(1, keepdim=True)[1]

                out_task_1 = np.append(out_task_1, pred_1.cpu().numpy())
                out_task_2 = np.append(out_task_2, pred_2.cpu().numpy())
                out_task_3 = np.append(out_task_3, pred_3.cpu().numpy())

                trg_task_1 = np.append(trg_task_1, label1.data.cpu().numpy())
                trg_task_2 = np.append(trg_task_2, label2.data.cpu().numpy())
                trg_task_3 = np.append(trg_task_3, label3.data.cpu().numpy())

    if (training_mode == "self_supervised") or (training_mode == "SupCon") or (training_mode == "self_supervised_decoder"):
        total_loss = 0
        total_acc = 0
        return total_loss, total_acc, [], [], [], [], [], []

    if training_mode == "ft_1p":
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, loss_task1, loss_task2, loss_task3, trg_task_1, trg_task_2, trg_task_3

    else:
        total_loss = torch.tensor(total_loss).mean()  # average loss
        total_acc = torch.tensor(total_acc).mean()  # average acc
        return total_loss, total_acc, out_task_1, out_task_2, out_task_3, trg_task_1, trg_task_2, trg_task_3

def model_evaluate_generateZ(model, temporal_contr_model, test_dl, device, training_mode):
    model.eval()
    temporal_contr_model.eval()

    total_loss = []
    total_acc = []

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, label1, label2, label3, _, _ in test_dl:
            data, label1, label2, label3 = data.float().to(device), label1.float().long().to(device), label2.float().long().to(device), label3.float().long().to(device)
            # print(label1.shape)
            # print(label2.shape)

            output = model(data)

            # compute loss
            if (training_mode != "self_supervised") and (training_mode != "SupCon"):
                prediction_task_1, prediction_task_2, prediction_task_3, features = output
                # print(prediction_task_1.shape)
                loss_task1_next = criterion(prediction_task_1, label1)
                loss_task2_next = criterion(prediction_task_2, label2)
                loss_task3_next = criterion(prediction_task_3, label3)


    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    return total_loss, total_acc, loss_task1_next, loss_task2_next, loss_task3_next



def gen_pseudo_labels(model, dataloader, device, experiment_log_dir):
    from sklearn.metrics import accuracy_score
    model.eval()
    softmax = nn.Softmax(dim=1)

    # saving output data
    all_pseudo_label1 = np.array([])
    all_pseudo_label2 = np.array([])
    all_pseudo_label3 = np.array([])

    all_label1 = np.array([])
    all_label2 = np.array([])
    all_label3 = np.array([])

    all_data = []

    with torch.no_grad():
        for data, label1, label2, label3, _, _ in dataloader:
            data = data.float().to(device)
            label1 = label1.view((-1)).long().to(device)
            label2 = label2.view((-1)).long().to(device)
            label3 = label3.view((-1)).long().to(device)

            # forward pass
            prediction_task_1, prediction_task_2, prediction_task_3, features = model(data)


            normalized_pred_1 = softmax(prediction_task_1)
            normalized_pred_2 = softmax(prediction_task_2)
            normalized_pred_3 = softmax(prediction_task_3)

            pseudo_label1 = normalized_pred_1.max(1, keepdim=True)[1].squeeze()
            pseudo_label2 = normalized_pred_2.max(1, keepdim=True)[1].squeeze()
            pseudo_label3 = normalized_pred_3.max(1, keepdim=True)[1].squeeze()

            all_pseudo_label1 = np.append(all_pseudo_label1, pseudo_label1.cpu().numpy())
            all_pseudo_label2 = np.append(all_pseudo_label2, pseudo_label2.cpu().numpy())
            all_pseudo_label3 = np.append(all_pseudo_label3, pseudo_label3.cpu().numpy())

            all_label1 = np.append(all_label1, label1.cpu().numpy())
            all_label2 = np.append(all_label2, label2.cpu().numpy())
            all_label3 = np.append(all_label3, label3.cpu().numpy())
            all_data.append(data)

    all_data = torch.cat(all_data, dim=0)

    data_save = dict()
    data_save["samples"] = all_data
    data_save["labels_task300"] = torch.LongTensor(torch.from_numpy(all_pseudo_label1).long())
    data_save["labels_task450"] = torch.LongTensor(torch.from_numpy(all_pseudo_label2).long())
    data_save["labels_task4704"] = torch.LongTensor(torch.from_numpy(all_pseudo_label3).long())
    file_name = f"pseudo_train_data.pt"
    torch.save(data_save, os.path.join(experiment_log_dir, file_name))
    print("Pseudo labels generated ...")
