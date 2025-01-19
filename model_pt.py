import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import numpy as np
import pandas as pd
from skimage.morphology import skeletonize
from scipy.spatial import distance_matrix
from skimage.draw import line_nd
from skimage.measure import label as sk_label

import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timeit import default_timer as timer

from dataset_pt import ODT3D_VT_Dataset, ODT3D_VT_Dataset_Val
from dataset_pt_MSD import MSD3D_VT_Dataset, MSD3D_VT_Dataset_Val, collate_batch
from utils import get_config_from_json, DiceLoss, MaskedL1Loss
from utils import ConnectionMetric, ConnLoss, MaskedDiceLoss

#from backbone.highresnet import HighRes3DNet
from backbone.unet3d import UNet_ODT, UNet_ODT_wCLS, UNet_ODT_wCLS_wEDT, UNet_ODT_wEDT


def train_model(config, model_name):
    save_path = os.path.join('./weights', model_name)
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # Training setting.
    num_epoch = config['epochs']
    lr = config['learning_rate']
    skelloss_weight = config['skelloss_weight']
    clsloss_weight = config['clsloss_weight']
    bgmean_weight = config['bgmean_weight']
    save_period = config['save_period']
    device = torch.device("cuda:0")
    # Enables benchmark mode in cudnn, which is good whenever your
    # input sizes for your network do not vary.
    torch.backends.cudnn.benchmark = False # True # Maybe this is the reason for random output.

    # Prepare data.
    if config['dataset'] == 'ODT':
        dataset_train = ODT3D_VT_Dataset(config)#, debug=True)
        train_gen = DataLoader(dataset_train, batch_size=None, shuffle=False, num_workers=4)
        dataset_val = ODT3D_VT_Dataset_Val(config)
        val_gen = DataLoader(dataset_val, batch_size=None, shuffle=False, num_workers=4)
    else:
        dataset_train = MSD3D_VT_Dataset(config)#, debug=True)
        train_gen = DataLoader(
            dataset_train, batch_size=config['sample_per_batch'],
            collate_fn=collate_batch, shuffle=False, num_workers=4)
        dataset_val = MSD3D_VT_Dataset_Val(config)
        val_gen = DataLoader(
            dataset_val, batch_size=config['sample_per_batch'],
            collate_fn=collate_batch, shuffle=False, num_workers=4)

    # Model initialization.
    #model = HighRes3DNet(in_channels=1, out_channels=1)
    model = UNet_ODT() # [32, 64, 64, 128, 256]
    #model = UNet_ODT(feat_channels=[32, 32, 32, 64, 128]) # chhalf
    #model = UNet_ODT(feat_channels=[64, 128, 128, 256, 512]) #chdouble

    #model = UNet_ODT_wCLS()
    #model = UNet_ODT_wEDT(num_channels=2) # with EDT only, no CLS.
    # [2, 3] for CLS; [1, 3] for CLS+Seg; [2, 4] for CLS+EDT
    [start_channel, end_channel] = [2, 3]

    model = model.to(device)
    print('-----------model-----------\n', model, '\n-----------end-model-----------\n')

    loss_fn_skel_val = ConnLoss()
    loss_fn_skel_train = ConnLoss()
    loss_fn_seg = MaskedDiceLoss()
    loss_fn_reg = MaskedL1Loss()
    # More numerically stable because of the log-sum-exp trick.
    loss_fn_cls = nn.BCEWithLogitsLoss(reduction='none')

    metric_fn = ConnectionMetric()


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5,
        patience=20, cooldown=5, min_lr=1e-6, eps=1e-08, verbose=True)
    #file_writer = SummaryWriter(log_dir=save_path + '/metrics')
    file_writer = SummaryWriter(log_dir=save_path)

    # Constructs scaler once.
    # GradScaler helps prevent gradients “underflowing”.
    scaler = torch.cuda.amp.GradScaler()

    # Training/val.
    best_val_loss = 99999.
    train_recorder_list = []
    val_recorder_list = []
    for epoch in range(1, num_epoch + 1):
        start_time = timer()
        # Reset metrics.
        train_recorder = []
        val_recorder = []

        # Training.
        model.train()
        idx_batch = 0
        for X, roi_mask, X_GT, in train_gen:

            # Only skel.
            X = X[:, start_channel:end_channel] 
            X_GT = X_GT[:, start_channel:end_channel]

            X = X.to(device)
            roi_mask = roi_mask.to(device)
            X_GT = X_GT.to(device)
            #Y = Y.to(device)

            with torch.cuda.amp.autocast():
                optimizer.zero_grad()
                output_skel = model(X)
                #output_skel, output_cls = model(X)
                #output_skel, output_reg = model(X)
                #output_skel, output_seg = model(X)

                # Keep the channel-dim (even if only 1) of GT for loss compatibility.
                loss_connection, loss_naive = loss_fn_skel_train(output_skel, X_GT[:, :1], roi_mask)
                loss_skel_pos = skelloss_weight * (loss_connection + bgmean_weight*loss_naive)

                recall, success_rate = metric_fn(output_skel, X_GT[:, :1], roi_mask)

                #loss_cls = loss_fn_cls(output_cls, )

                #loss_reg_fg, loss_reg_nonfg = loss_fn_reg(output_reg, X_GT[:, 1:], roi_mask)
                #loss_reg = loss_reg_fg + loss_reg_nonfg

                #loss_seg_full, loss_seg_roi = loss_fn_seg(output_seg, X_GT[:, :1], roi_mask)
                #loss_seg = loss_seg_full + loss_seg_roi

                loss = loss_skel_pos
                #loss = loss_skel_pos + loss_reg
                #loss = loss_skel_pos + loss_seg_roi

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_recorder.append([
                loss_skel_pos.item(), recall.item(), success_rate.item(),
                #loss_reg.item(), loss_reg_fg.item(), loss_reg_nonfg.item(),
                #loss_seg.item(), loss_seg_full.item(), loss_seg_roi.item(),
                loss_connection.item(), loss_naive.item()])

            end_time_trainbatch = timer()
            idx_batch += 1
        end_time_train = timer()

        # Validation.
        model.eval()
        for X, roi_mask, X_GT, in val_gen:
            # Only skel.
            X = X[:, start_channel:end_channel]
            X_GT = X_GT[:, start_channel:end_channel]

            X = X.to(device)
            roi_mask = roi_mask.to(device)
            X_GT = X_GT.to(device)

            with torch.cuda.amp.autocast():
                output_skel = model(X)
                #output_skel, output_reg = model(X)
                #output_skel, output_seg = model(X)

                # Still use original loss_fn and fixed valset.
                loss_connection, loss_naive = loss_fn_skel_train(output_skel, X_GT[:, :1], roi_mask)
                loss_skel_pos = skelloss_weight * (loss_connection + bgmean_weight*loss_naive)
                loss_connection_val, loss_naive_val = loss_fn_skel_val(output_skel, X_GT[:, :1], roi_mask)
                loss_skel_pos_val = skelloss_weight * (loss_connection_val + loss_naive_val)
                
                recall, success_rate = metric_fn(output_skel, X_GT[:, :1], roi_mask)

                #loss_reg_fg, loss_reg_nonfg = loss_fn_reg(output_reg, X_GT[:, 1:], roi_mask)
                #loss_reg = loss_reg_fg + loss_reg_nonfg

                #loss_seg_full, loss_seg_roi = loss_fn_seg(output_seg, X_GT[:, :1], roi_mask)
                #loss_seg = loss_seg_full + loss_seg_roi


                val_recorder.append([
                    loss_skel_pos.item(), recall.item(), success_rate.item(),
                    #loss_reg.item(), loss_reg_fg.item(), loss_reg_nonfg.item(),
                    #loss_seg.item(), loss_seg_full.item(), loss_seg_roi.item(),
                    loss_skel_pos_val.item(), loss_connection_val.item(), loss_naive_val.item(),
                    loss_connection.item(), loss_naive.item()])

        train_recorder = np.array(train_recorder).sum(0) / len(train_gen)
        val_recorder = np.array(val_recorder).sum(0) / len(val_gen)
        end_time = timer()

        # lr_scheduler is still based on the main task: skel completion.
        # loss_skel_pos_val: recorder[-5] x3.
        scheduler.step(val_recorder[-5])
        #better = val_loss_skel < best_val_loss
        better = val_recorder[-5] < best_val_loss

        print(
            'Ep-{:03d}|Train>Skel:{:.04f}(={:.04f}+{:.04f}),'.format(
                epoch, train_recorder[0], train_recorder[-2], train_recorder[-1]),
            #'Reg:{:.04f}(={:.04f}+{:.04f})'.format(*train_recorder[3:6]),
            #'Seg:{:.04f}(={:.04f}+{:.04f})'.format(*train_recorder[3:6]),
            #'Recall(SR): {:.06f}({:.06f})'.format(train_recorder[1], train_recorder[2]),
            '|Val>Skel:{:.04f}(={:.04f}+{:.04f}),'.format(val_recorder[0], val_recorder[-2], val_recorder[-1]),
            'Skel_val: {:.04f}(={:.04f}+{:.04f})'.format(*val_recorder[-5:-2]),
            #'Reg:{:.04f}(={:.04f}+{:.04f})'.format(*val_recorder[3:6]),
            #'Seg:{:.04f}(={:.04f}+{:.04f})'.format(*val_recorder[3:6]),
            #'Recall(SR): {:.04f}({:.04f})'.format(val_recorder[1], val_recorder[2]),
            ' {:.04f}({:.04f})'.format(val_recorder[1], val_recorder[2]),
            '|{:.01f}s {}'.format(end_time - start_time, '[Up]' if better else ''))

        # Checkpoint periodly.
        if (epoch % save_period) == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'model_weights_ep{:03d}.pth'.format(epoch)))
            #model.load_state_dict(torch.load('model_weights.pth'))

        # Checkpoint if improved.
        if better:
            #print('\t-->Better val_loss_skel: {:.04f}@ep{:04d}.'.format(val_loss_skel, epoch))
            torch.save(model.state_dict(), os.path.join(save_path, 'model_weights_best.pth'))
            best_val_loss = val_recorder[-5]
        '''
        # Testing.
        if better or ((epoch%save_period) == 0):
            model.eval()
            acc = []
            acc_patch = []
            for idx_sample in range(len(dataset_val)):
                idx_image = np.int32(dataset_val.imglist_val[idx_sample].split('.npy')[0][-3:])
                df_cur = dataset_val.df_all.loc[dataset_val.df_all.img_id == idx_image]
                batch_size_cur = len(df_cur)
                
                # idx-37,60, eps too far, drop
                if (idx_sample in [37, 60]) or (batch_size_cur == 0):
                    #print('idx-{:03d} | idx_image-{:03d} has no APs. Skip.'.format(idx_sample, idx_image))
                    continue

                # Load data.
                image_fused, skeleton_skan, branch_data = dataset_val.datalist[idx_sample]
                img_name = os.path.basename(dataset_val.imglist_val[idx_sample])
                
                sample_coord_testing = df_cur.iloc[:,1:4].values
                ep_pair_array_gt = df_cur.iloc[
                    :, dataset_val.coord_startidx:dataset_val.coord_startidx+6].values.reshape((batch_size_cur, 2, 3))
                ep_pair_array_A = df_cur.iloc[
                    :, dataset_val.coord_startidx+6:dataset_val.coord_startidx+12].values.reshape((batch_size_cur, 2, 3))
                
                # Prepare GT and automatic (A) prediction.
                ## Discretized straight line between the two EPs.
                ep_array_disc_A = [] # (z,y,x) in original and patch coordinate.
                ep_array_disc_gt = []
                for idx_patch, [[p_start_A, p_end_A], [p_start_gt, p_end_gt]] in enumerate(
                        zip(ep_pair_array_A, ep_pair_array_gt)):
                    ep_array_disc_A_cur = np.array(line_nd(p_start_A, p_end_A, endpoint=True, integer=True)).T
                    ep_array_disc_gt_cur = np.array(line_nd(p_start_gt, p_end_gt, endpoint=True, integer=True)).T
                    ep_array_disc_A.append([
                        ep_array_disc_A_cur,
                        ep_array_disc_A_cur + dataset_val.psize_half - sample_coord_testing[idx_patch]])
                    ep_array_disc_gt.append([
                        ep_array_disc_gt_cur,
                        ep_array_disc_gt_cur + dataset_val.psize_half - sample_coord_testing[idx_patch]])
                    
                # Prepare batch testing data.
                image_fused_patch = np.zeros(
                    (batch_size_cur, dataset_val.c_num, dataset_val.patch_z, dataset_val.patch_y, dataset_val.patch_x),
                    dtype=dataset_val.datatype)
                for idx_patch in range(batch_size_cur):
                    # (z,y,x) for AP.
                    zyx = sample_coord_testing[idx_patch]
                    image_fused_patch[idx_patch] = image_fused[
                        :,
                        zyx[0] - dataset_val.psize_half[0] : zyx[0] + dataset_val.psize_half[0],
                        zyx[1] - dataset_val.psize_half[1] : zyx[1] + dataset_val.psize_half[1],
                        zyx[2] - dataset_val.psize_half[2] : zyx[2] + dataset_val.psize_half[2]].copy()

                # Inference.
                with torch.no_grad():
                    X_input = torch.from_numpy(image_fused_patch[:, 2:3]).to(device)

                    with torch.cuda.amp.autocast():
                        output_skel = model(X_input)
                    output_skel = output_skel.detach().cpu().numpy()

                # Post-processing for prediction.
                ## Must be bool because True+True==True.
                mask_pred_patch = image_fused_patch[:,1:2].astype(np.bool8) + (output_skel > 0.5)
                mask_pred = image_fused[1].copy().astype(np.bool8)
                for idx_patch in range(batch_size_cur):
                    zyx = sample_coord_testing[idx_patch] # (z,y,x) for AP.
                    mask_pred[
                        zyx[0] - dataset_val.psize_half[0] : zyx[0] + dataset_val.psize_half[0],
                        zyx[1] - dataset_val.psize_half[1] : zyx[1] + dataset_val.psize_half[1],
                        zyx[2] - dataset_val.psize_half[2] : zyx[2] + dataset_val.psize_half[2]] += mask_pred_patch[
                            idx_patch, 0]

                skel_pred = skeletonize(mask_pred)
                mask_pred_sep = sk_label(mask_pred)
                skel_pred_sep = sk_label(skel_pred)

                # Calculate accuracy.
                acc_cur = 0
                for idx_patch in range(batch_size_cur):
                    # skel_pred_sep
                    predlabel_start = mask_pred_sep[
                        ep_pair_array_gt[idx_patch][0, 0],
                        ep_pair_array_gt[idx_patch][0, 1],
                        ep_pair_array_gt[idx_patch][0, 2]]
                    predlabel_end = mask_pred_sep[
                        ep_pair_array_gt[idx_patch][1, 0],
                        ep_pair_array_gt[idx_patch][1, 1],
                        ep_pair_array_gt[idx_patch][1, 2]]

                    acc_cur += (predlabel_start) > 0 and (predlabel_start == predlabel_end)

                acc.append([idx_image, acc_cur, batch_size_cur, float(acc_cur)/batch_size_cur])
                
                # Calculate patch-based accuracy.
                acc_patch_cur = 0
                TP, FN, FP, TN, FP_dual = [0, 0, 0, 0, 0]
                for idx_patch in range(batch_size_cur):
                    ep_pair_array_A_patch = dataset_val.coordconvert_full2patch(
                        ep_pair_array_A[idx_patch], sample_coord_testing[idx_patch])
                    ep_pair_array_gt_patch = dataset_val.coordconvert_full2patch(
                        ep_pair_array_gt[idx_patch], sample_coord_testing[idx_patch])

                    mask_pred_patch_sep = sk_label(mask_pred_patch[idx_patch, 0])
                    
                    if dataset_val.is_trueconnection(df_cur.iloc[idx_patch].label):
                        if dataset_val.is_connected(mask_pred_patch_sep, ep_pair_array_gt_patch):
                            TP += 1
                        else:
                            FN += 1
                        if np.any(ep_pair_array_A_patch != ep_pair_array_gt_patch):
                            # If A-pair is not GT-pair and is connected by model, count as FP_dual.
                            FP_dual += dataset_val.is_connected(mask_pred_patch_sep, ep_pair_array_A_patch)
                    else:
                        # No GT-pair. Only evaluate A-pair.
                        if dataset_val.is_connected(mask_pred_patch_sep, ep_pair_array_A_patch):
                            FP += 1
                            FP_dual += 1
                        else:
                            TN += 1

                acc_patch.append([idx_image, TP, FN, FP, TN, FP_dual, batch_size_cur])
                
            acc_patch = np.array(acc_patch)
            dataset_val.eval_report(acc_patch)

            '''

        train_recorder_list.append(train_recorder)
        val_recorder_list.append(val_recorder)

    train_recorder_list = np.array(train_recorder_list)
    val_recorder_list = np.array(val_recorder_list)

    print('Training done.\nBest val_loss: {:.04f}@Epoch-{:03d}.'.format(
        np.min(val_recorder_list[:,0]), 1 + np.argmin(val_recorder_list[:,0])))
    print('Best recall: {:.04f}@Epoch-{:03d}.'.format(
        np.min(val_recorder_list[:,1]), 1 + np.argmin(val_recorder_list[:,1])))
    print('Best success_rate: {:.04f}@Epoch-{:03d}.'.format(
        np.min(val_recorder_list[:,2]), 1 + np.argmin(val_recorder_list[:,2])))

    data_array = np.concatenate((train_recorder_list, val_recorder_list), axis=1)
    df = pd.DataFrame(data_array, columns=[
        'train_skel','train_recallSP','train_srSP',
        #'train_reg', 'train_reg_fg', 'train_reg_nonfg',
        'train_conn', 'train_naive',
        'val_skel','val_recallSP','val_srSP',
        #'val_reg', 'val_reg_fg', 'val_reg_nonfg',
        'Val_skel', 'Val_conn', 'Val_naive',
        'val_conn', 'val_naive'])
    df.to_csv(os.path.join(save_path, 'training_log.csv'))


if __name__ == '__main__':
    config = get_config_from_json('config_unet.json')
    #config = get_config_from_json('config_unet_MSD.json')
    #config = get_config_from_json('config_unet_MSD_nnUNet.json')
    print('-----------config-----------\n', config, '\n-----------end-config-----------\n')
    #model_name = 'odt3d_vt_unet_bz128_prototype_11'
    model_name = config['model_name']
    train_model(config, model_name)

