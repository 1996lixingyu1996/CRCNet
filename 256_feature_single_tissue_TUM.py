import gc
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from survival_metric import *
import argparse
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from nature_method_model import TopBottomMethod
from torch.optim.lr_scheduler import ReduceLROnPlateau
from early_stopping import EarlyStopping
from dataloader_prob_mean_ver01 import *
import os
import glob

torch.multiprocessing.set_sharing_strategy('file_system')
import time



os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def _neg_partial_log(prediction, T, E):
    """
     calculate cox loss
    :param X: variables
    :param T: Time
    :param E: Status  0:censored data  1: occurred data
    :return: neg log of the likelihood
    """

    current_batch_len = len(prediction)

    R_matrix_train = np.zeros([current_batch_len, current_batch_len], dtype=int)

    for i in range(current_batch_len):
        for j in range(current_batch_len):
            R_matrix_train[i, j] = T[j] >= T[i]

    train_R = torch.FloatTensor(R_matrix_train)
    train_R = train_R.cuda()
    train_ystatus = torch.FloatTensor(E).cuda()

    theta = prediction.reshape(-1)
    exp_theta = torch.exp(theta)

    loss_nn = - torch.mean((theta - torch.log(torch.sum(exp_theta * train_R, dim=1))) * train_ystatus)

    return loss_nn


def prediction(model, queryloader, device, l1_reg=None, l1_reg_coef=1e-5, testing=False):
    """

    :param model: pytorch model
    :param queryloader: dataloader
    :param testing:
    :return:
    """

    model.eval()

    lbl_pred_all = None
    status_all = []
    survtime_all = []
    iter = 0
    gc.collect()
    tbar = tqdm(queryloader)

    survival_time_torch = None
    lbl_torch = None

    with torch.no_grad():
        for i_batch, sampled_batch in enumerate(tbar):

            feature, survival_time, status = sampled_batch["feature"], sampled_batch["survival_time"], sampled_batch[
                "survival_status"]

            graph = feature.to(device)
            status_device = status.to(device)
            survival_time_device = survival_time.to(device)
            time_cpu = survival_time.cpu().numpy()
            status_cpu = status_device.data.cpu().numpy()

            time_squeeze = np.squeeze(time_cpu)
            status_squeeze = np.squeeze(status_cpu)

            survtime_all.append(time_squeeze)
            status_all.append(status_squeeze)

            # ============forward=================

            lbl_pred = model(graph)

            """
                        if i_batch == 0:
                lbl_pred_all = lbl_pred
                survival_time_torch = survival_time
                lbl_torch = status_device

            """

            if iter == 0:

                lbl_pred_all = lbl_pred
                survival_time_torch = survival_time_device
                lbl_torch = status_device
            else:
                lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
                # print("survival_time_torch:", survival_time_torch)
                survival_time_torch = torch.cat([survival_time_torch, survival_time_device])
                # print("lbl_torch_type:", lbl_torch.shape)
                lbl_torch = torch.cat([lbl_torch, status_device])

            iter += 1

    survtime_all = np.asarray(survtime_all)
    status_all = np.asarray(status_all)

    loss_surv = _neg_partial_log(lbl_pred_all, survtime_all, status_all)
    loss = loss_surv

    pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survival_time_torch)
    c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survival_time_torch)

    if not testing:
        print('\n[val]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))
    else:
        print('\n[testing]\t loss (nn):{:.4f}'.format(loss.data.item()),
              'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))

    return loss.data.item(), c_index


def train_epoch(epoch, model, optimizer, trainloader, device, batch_size, measure=1, verbose=1):
    model.train()

    lbl_pred_all = None
    lbl_pred_each = None

    survtime_all = []
    status_all = []

    iter = 0
    gc.collect()
    loss_nn_all = []

    tbar = tqdm(trainloader)

    for i_batch, sampled_batch in enumerate(tbar):

        feature, survival_time, status = sampled_batch["feature"], sampled_batch["survival_time"], sampled_batch[
            "survival_status"]

        graph = feature.to(device)
        status_device = status.to(device)
        survival_time_device = survival_time.to(device)

        lbl_pred = model(graph)

        time_cpu = survival_time.cpu().numpy()
        status_cpu = status.cpu().numpy()

        time_squeeze = np.squeeze(time_cpu)
        status_squeeze = np.squeeze(status_cpu)

        survtime_all.append(time_squeeze)
        status_all.append(status_squeeze)

        if i_batch == 0:
            lbl_pred_all = lbl_pred
            survival_time_torch = survival_time_device
            lbl_torch = status_device

        if iter == 0:
            lbl_pred_each = lbl_pred

        else:
            lbl_pred_all = torch.cat([lbl_pred_all, lbl_pred])
            lbl_pred_each = torch.cat([lbl_pred_each, lbl_pred])

            lbl_torch = torch.cat([lbl_torch, status_device])
            survival_time_torch = torch.cat([survival_time_torch, survival_time_device])

        iter += 1

        if iter % 16 == 0 or i_batch == len(trainloader) - 1:
            # update the loss when collect 16 data samples

            survtime_all = np.asarray(survtime_all)
            status_all = np.asarray(status_all)

            # print(survtime_all)

            if np.max(status_all) == 0:
                print("encounter no death in a batch, skip")
                lbl_pred_each = None
                survtime_all = []
                status_all = []
                iter = 0
                continue

            optimizer.zero_grad()  # zero the gradient buffer

            loss_surv = _neg_partial_log(lbl_pred_each, survtime_all, status_all)
            """
                                l1_reg = None
                for W in model.parameters():
                    if l1_reg is None:
                        l1_reg = torch.abs(W).sum()
                else:
                    l1_reg = l1_reg + torch.abs(W).sum()  # torch.abs(W).sum() is equivalent to W.norm(1)

                loss = loss_surv + 1e-5 * l1_reg

            """
            loss = loss_surv
            # ===================backward====================
            loss.backward()
            optimizer.step()

            torch.cuda.empty_cache()
            lbl_pred_each = None
            survtime_all = []
            status_all = []
            loss_nn_all.append(loss.data.item())
            iter = 0

            gc.collect()

    if measure:
        pvalue_pred = cox_log_rank(lbl_pred_all.data, lbl_torch, survival_time_torch)
        c_index = CIndex_lifeline(lbl_pred_all.data, lbl_torch, survival_time_torch)

        if verbose > 0:
            print("\nEpoch: {}, loss_nn: {}".format(epoch, np.mean(loss_nn_all)))
            print('\n[Training]\t loss (nn):{:.4f}'.format(np.mean(loss_nn_all)),
                  'c_index: {:.4f}, p-value: {:.3e}'.format(c_index, pvalue_pred))


def train(train_path, test_path, model_save_path, num_epochs, lr, device, patch_number, feature_channels, \
          top_bottom_k=10, weight_decay=5e-4):
    model = TopBottomMethod(patch_number=patch_number, feature_channels=feature_channels, \
                                 top_bottom_k=top_bottom_k).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    Data = My_dataloader(data_path=train_path, batch_size=batch_size,
                         train=True)  # MIL_dataloader(data_path=train_path, cluster_num = cluster_num, train=True)
    # print("True, trainloader!")
    trainloader, valloader = Data.get_loader()  # Data.get_loader()

    TestData = My_dataloader(data_path=test_path, batch_size=batch_size,
                             train=False)  # MIL_dataloader(test_path, cluster_num=cluster_num, train=False)
    testloader = TestData.get_loader()

    # initialize the early_stopping object
    early_stopping = EarlyStopping(model_path=model_save_path, patience=20, verbose=True)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)
    # print("num_epochs:", num_epochs)
    save_epoch = range(10, int(num_epochs), 5)
    val_ci_list = []
    val_losses = []

    for epoch in range(int(num_epochs)):
        print("device", device)
        train_epoch(epoch, model, optimizer, trainloader, device=device, batch_size=batch_size)
        valid_loss, val_ci = prediction(model, valloader, device=device)
        scheduler.step(valid_loss)
        val_losses.append(valid_loss)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        if epoch in save_epoch:
            val_ci_list.append(val_ci)
            print('saving epoch in {}, vali loss: {}, val ci:{}'.format(epoch, valid_loss, val_ci))
            torch.save(model, model_save_path)  # save whole model rather than parameter

    # Use the final saved model to test this time
    model_test = torch.load(model_save_path)

    _, c_index_test = prediction(model_test, testloader, testing=True, device=device)
    _, c_index_validation = prediction(model_test, valloader, testing=True, device=device)
    _, c_index_train = prediction(model_test, trainloader, testing=True, device=device)
    return c_index_train, c_index_validation, c_index_test, val_ci_list, val_losses


if __name__ == '__main__':

    # args = parser.parse_args()
    time_start = time.time()

    """
    parser = argparse.ArgumentParser(description="Probability model parameter")
    parser.add_argument("--nepochs", type=int, default=100, help="The maxium number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("feature_folder_path", type=str, default="", help="feature save path")
    """

    patch_number = 
    tissue_name = "TUM"

    MCO_feature_path = ""
    TCGA_feature_path = ""

    feature_folder_path = TCGA_feature_path  
    param_excel = pd.read_excel(parameter_excel_path, engine="openpyxl")
    # param_index = 0  # loop parameter index
    param_number = len(param_excel)
    # train(train_path, test_path, model_save_path, num_epochs, lr, device, patch_number, feature_channels, \
    #           top_bottom_k=10, weight_decay=5e-4)
    model_save_folder_path = "/data/lxy/TCGA_model/256feature_model_single_0526_" + tissue_name # "/data/lxy/MCO_model/256feature_model_single_0511" + tissue_name
    if not os.path.exists(model_save_folder_path):
        os.mkdir(model_save_folder_path)

    for i in range(1, param_number + 1, 1):

        param_info = param_excel.iloc[i - 1]
        batch_size = param_info["batch_size"]  #
        num_epochs = param_info["epochs"]  #
        lr = param_info["lr"]  #
        weight_decay = param_info["weight_decay"]

        feature_paths = glob.glob(feature_folder_path + "/*.npz")

        top_bottom_k = 10  #
        # detect device
        device = torch.device("cpu")
        if (torch.cuda.is_available()):
            device = torch.device("cuda")

        feature_stage = []
        feature_channel = 256  # probability model is 8 , nature medicine method is 256
        for feature_path in feature_paths:
            feature_composite = np.load(feature_path)
            stage = feature_composite["stage"]
            feature_stage.append(stage)

        feature_index = range(len(feature_paths))

        kf = StratifiedKFold(n_splits=5)

        fold = 0

        # cross validation
        for train_index, test_index in kf.split(feature_index, feature_stage):
            print("Now training fold:{}".format(fold))

            test_id = [feature_paths[i] for i in test_index]
            train_id = [feature_paths[j] for j in train_index]

            model_save_path = model_save_folder_path + "/" + "model_param_index" + str(i) + "_fold_" + str(
                fold) + ".pth"
            # '/data/lxy/HE2RNA_binary_model/model_fold_{}.pth'.format(fold)
            """
            train(train_path, test_path, model_save_path, num_epochs, lr, device, patch_number, feature_channels, \
            top_bottom_k=10, weight_decay=5e-4):
            """
            # return c_index_train, c_index_validation, c_index_test, val_ci_list, val_losses
            c_index_train, c_index_validation, c_index_test, val_ci_list, val_losses = train(train_path=train_id,
                                                                                             test_path=test_id,
                                                                                             model_save_path=model_save_path,
                                                                                             num_epochs=num_epochs,
                                                                                             lr=lr, \
                                                                                             device=device,
                                                                                             patch_number=patch_number,
                                                                                             feature_channels=feature_channel,
                                                                                             top_bottom_k=top_bottom_k,
                                                                                             weight_decay=weight_decay)
            loss_cindex_data = {"val_ci": val_ci_list, "val_losses": val_losses}

            param_excel.iloc[i - 1, (training_cindex_index + fold)] = c_index_train
            param_excel.iloc[i - 1, (validation_cindex_index + fold)] = c_index_validation
            param_excel.iloc[i - 1, (testing_cindex_index + fold)] = c_index_test
            param_excel.iloc[i - 1, 14] = model_save_path
            loss_cindex_df = pd.DataFrame(loss_cindex_data, columns={"val_ci", "val_loss"})
            loss_cindex_df_save_path = model_save_folder_path + "/" + "model_param_info_index_" + str(
                i) + "_fold_" + str(fold) + ".csv"
            print("sava_path:", loss_cindex_df_save_path)
            loss_cindex_df.to_csv(loss_cindex_df_save_path)

            fold += 1

    parameter_csv_save_path = model_save_folder_path + "/" + "composite_info" + ".csv"
    param_excel.to_csv(parameter_csv_save_path)

    time_end = time.time()
    print("total_time:", time_end - time_start)

