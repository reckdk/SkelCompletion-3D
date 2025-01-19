import os
import random
import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from tqdm import tqdm
from scipy.ndimage import binary_dilation, generate_binary_structure
from scipy.ndimage import distance_transform_edt
from glob import glob
#import cv2
import skimage.io as skio
import torch
#import h5py
from skan import Skeleton as skan_Skeleton
from skan import summarize as skan_summarize


# DataLoader(dataset_train, batch_size=None, shuffle=False, num_workers=4)
# For future multiple GPUs training.
def collate_batch(batchdata):
    image_fused_patch = batchdata[0][0].copy()
    roi_mask_patch = batchdata[0][1].copy()
    image_fused_patch_gt = batchdata[0][2].copy()
    for image_fused_patch_cur, roi_mask_patch_cur, image_fused_patch_gt_cur in batchdata[1:]:
        image_fused_patch = np.concatenate((image_fused_patch, image_fused_patch_cur), axis=0)
        roi_mask_patch = np.concatenate((roi_mask_patch, roi_mask_patch_cur), axis=0)
        image_fused_patch_gt = np.concatenate((image_fused_patch_gt, image_fused_patch_gt_cur), axis=0)

    return (torch.from_numpy(image_fused_patch),
            torch.from_numpy(roi_mask_patch),
            torch.from_numpy(image_fused_patch_gt))
'''
        if deterministic:
            np.random.seed(12345)
            torch.manual_seed(12345)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(12345)
            cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
'''

class MSD3D_VT_Dataset(torch.utils.data.Dataset):
    def __init__(self, config, training=True, debug=False, shuffle=True):
        self.training = training
        self.debug = debug
        self.shuffle = shuffle
        self.dataset_dir = config['dataset_dir']
        self.datatype = np.float32
        self.CLS_branch = config['CLS_branch']
        self.SEG_branch = config['SEG_branch']
        self.EDT_branch = config['EDT_branch']
        self.c_num = 4 if self.EDT_branch else 3 # Fuse EDT or not.
        self.batch_size = config['batch_size']
        assert self.batch_size%4 == 0,\
            'Batchsize mush be 4 or multiples of 4.'
        self.batch_size_p = self.batch_size>>1 # Half sampels are positive.
        self.batch_size_3over4 = self.batch_size_p + (self.batch_size_p>>1)
        # # (N,H,W) or axis (z,y,x). Each size must be an even number.
        self.patch_size = np.array(config['patch_size'])
        self.psize_half = self.patch_size >> 1
        [self.patch_z, self.patch_y, self.patch_x] = config['patch_size']
        #[self.patch_z_half, self.patch_y_half, self.patch_x_half]  = self.psize_half

        # patch_y_bottom

        # Augmentation initializaiton.
        self.aug_intensity = True
        self.aug_flip = True
        ## Flip flags. [2,3,4] --> [z,y,x].
        self.flip_op = [[], [2], [3], [4], [2, 3], [2, 4], [3, 4], [2, 3, 4]]

        # AP related hyper paremeters.
        # Assume the path points 15 away from the end points may be available.
        self.IDX_TO_EP = 5 # 15 # [Need finetune].
        self.IDX_SHIFT = 7 # 0 # 7 # [Need finetune].
        #self.DISTANCE_MAX = int(branch_data_long['branch-distance'].max())
        self.DISTANCE_MAX = 2048
        assert self.DISTANCE_MAX > 2*self.IDX_TO_EP,\
            'There is no branches longer enough. Please check your data.'
        # Duplicate sampling ratio. To extract enough candidates.
        self.DUP_SAMPLING_RATIO = 5

        # RoI BBox related hyper paremeters.
        self.SKEL_LEN_TH = 20 #50
        self.REMOVED_MAX = [2, 10] #[prev] [2, 8] # [1, 4] # [3, 9]  # [3, 12]
        self.binomp = 0.8 # Percentage of kept removed_idx. For multi-breaks.
        self.BBOX_SHIFT_MAX = 3 #3 #5 [Need finetune].

        # EDT related parameters.
        self.EDT_TH = 7
        self.dilate_struct = generate_binary_structure(3,2)
        self.EXTRA_DILATE_NUM = 5

        #self.rng = np.random.default_rng(221019)
        self.rng = np.random.default_rng() # Random training.
        self.debug_seed = 221026
        self.eval_seed = 221111

        # TODO: debug why does join() not work.
        #self.imglist_train = glob(os.path.join(self.dataset_dir, 'train/') + '*.npy')
        self.imglist_train = glob(os.path.join(self.dataset_dir, 'train/*.npy'))
        self.imglist_train.sort()
        
        # ln -s /home/oct3090/codes/ODT_3D/dataset/bak/MSD_Task008_HepaticVessel_x10/val ./val
        # ln -s /home/oct3090/codes/ODT_3D/dataset/bak/MSD_Task008_HepaticVessel_x10/prednnUNet_allfolds/fold0train_fold0val ./val
        self.imglist_val = glob(os.path.join(self.dataset_dir, 'val/*.npy'))
        self.imglist_val.sort()

        self.df_all = pd.read_csv('../dataset/MSD_Task008_HepaticVessel/MSD_annotation.csv', index_col=0)
        self.coord_startidx = 5

        if 0:
            # Drop the samples with few potention APs.
            invalid_list = [28, 30, 66, 80, 82, 84, 127, 146, 150, 157, 175,
                            194, 234, 248, 266, 272, 282, 368, 437, 458]
            valset_keys = set(int(img_path.split('.npy')[0][-3:]) for img_path in self.imglist_val)
            for idx_invalid in invalid_list:
                if idx_invalid in valset_keys:
                    self.imglist_val.remove(
                        '../dataset/MSD_Task008_HepaticVessel/val/hepaticvessel_{:03d}.npy'.format(idx_invalid))
                else:
                    self.imglist_train.remove(
                        '../dataset/MSD_Task008_HepaticVessel/train/hepaticvessel_{:03d}.npy'.format(idx_invalid))

        # Due to data insufficiency, the samples in val and test are
        # same but evaluated differently.
        self.imglist_test = self.imglist_val

        if self.debug:
            self.imglist_train = [self.imglist_train[0], self.imglist_train[-1]]
            self.rng = np.random.default_rng(self.debug_seed)

        if self.training:
            # Load all data into RAM for efficiency. About 23 GB?
            self.datalist_train = []
            for img_path in tqdm(self.imglist_train):

                image_fused = np.load(img_path)
                skeleton_skan = skan_Skeleton(image_fused[2])
                branch_data = skan_summarize(skeleton_skan)
                # Normailization.
                image_fused = image_fused.astype(self.datatype)
                image_fused[0] = image_fused[0]/image_fused[0].max()

                self.datalist_train.append([
                    image_fused,
                    skeleton_skan,
                    branch_data])

            self.datalist = self.datalist_train
            
        self.on_epoch_end()

    def on_epoch_end(self):    
        if self.debug:
            self.rng = np.random.default_rng(self.debug_seed)
        if self.shuffle:
            # No RAM crisis. 25.3 µs. Seems only shuffle pointers.
            random.shuffle(self.datalist_train)

    def coordconvert_full2patch(self, coord_full, AP_coord_full):
        '''
        Convert coordinates `coord_full` into patch-coordinates.
        '''
        return coord_full - AP_coord_full + self.psize_half

    def coordconvert_patch2full(self, coord_patch, AP_coord_full):
        '''
        Convert `patch-coordinates` into fullfield coordinates.
        '''
        return coord_patch - self.psize_half + AP_coord_full
    
    def __len__(self):
        return len(self.datalist_train)

    def __getitem__(self, index):
        data = self.try_getitem(index)
        # In case the current sampling fails, do a resample.
        while data is None:
            index = (index+self.rng.integers(len(self.datalist))) % len(self.datalist)
            data = self.try_getitem(index)
        return data

    def try_getitem(self, index):
        try:
            image_fused, skeleton_skan, branch_data = self.datalist[index]
            img_d, img_h, img_w = image_fused.shape[-3:]

            # Placeholder.
            image_fused_patch_gt = np.zeros(
                (self.batch_size, self.c_num, self.patch_z, self.patch_y, self.patch_x),
                dtype=self.datatype)

            image_fused_patch = np.zeros(
                (self.batch_size, self.c_num, self.patch_z, self.patch_y, self.patch_x),
                dtype=self.datatype)

            # In roi_mask_patch, c0: used, c1: mask/edt, c2: skel.
            roi_mask_patch = np.zeros(
                (self.batch_size, 3, self.patch_z, self.patch_y, self.patch_x),
                dtype=self.datatype)

            # Prepare training samples.
            # Only train / evaluate on long branches.
            branch_data_long = branch_data[
                branch_data['branch-distance'] > self.SKEL_LEN_TH]

            # Placeholder for potential Anchor Points (APs).
            # AP should be away from the two End Points (EPs).
            # Because there could be missing skel beyond EP (due to noise).
            idx_APs = self.rng.integers(
                low=0, high=self.DISTANCE_MAX,
                size=(len(branch_data_long), self.DUP_SAMPLING_RATIO))
            branches_len = branch_data_long.index.map(
                lambda idx_branch: len(skeleton_skan.path(idx_branch))
                ).values.reshape((-1, 1))
            # Clip coordinates within the length of corresponding branch.
            idx_APs = np.mod(idx_APs, branches_len - 2*self.IDX_TO_EP)
            idx_APs += self.IDX_TO_EP # Shift self.IDX_TO_EP away from the start point.

            # Drop invalid APs.
            # List of AP coordinates: [z,y,x].
            sample_coord_all = [] 
            # List of AP infomation:
            # [branch_id, voxel sequence id (seq_id) in branch].
            sample_info_all = []
            for idx_sample, idx_branch in enumerate(branch_data_long.index):
                sample_coord = skeleton_skan.coordinates[
                    skeleton_skan.path(idx_branch)[idx_APs[idx_sample]]]
                # Infomation used for removing skel voxel as training sample.
                # [branch_idx, skel voxel idx in corresponding branch].
                sample_info = np.array((
                    idx_branch*np.ones(self.DUP_SAMPLING_RATIO),
                    idx_APs[idx_sample])).T
                # Coordinates of the starting point and ending point.
                coord_EPs = skeleton_skan.coordinates[
                    skeleton_skan.path(idx_branch)[
                    ::branches_len[idx_sample,0]-1]]
                
                # Filter-1: an AP is available if its both EPs are not within BBox.
                # AP-1: BBox based filter.
                idx_available = np.logical_and(
                    np.max(np.abs(sample_coord-coord_EPs[0]), axis=1) > self.psize_half[0],
                    np.max(np.abs(sample_coord-coord_EPs[1]), axis=1) > self.psize_half[0])
                if np.any(idx_available):
                    sample_coord_all.append(sample_coord[idx_available])
                    sample_info_all.append(sample_info[idx_available])
                
                # AP-2: Euclidean distance based filter.
                #dist2startpoint = np.linalg.norm(sample_coord - coord_EPs[0], axis=1)
                #dist2endpoint = np.linalg.norm(sample_coord - coord_EPs[1], axis=1)    
            
            # Stack together for efficiency.
            sample_coord_all = np.concatenate(sample_coord_all, axis=0)
            sample_info_all = np.concatenate(sample_info_all, axis=0)
            # (d, h, w, branch_id, seq_id)
            sample_coord_all = np.concatenate((
                sample_coord_all, sample_info_all), axis=1).astype(np.int64)

            # Filter-2: remove the APs across the boundary.
            # BBOX_SHIFT_MAX is for shift aug.
            sample_coord_all = sample_coord_all[np.all(np.logical_and(
                sample_coord_all[:,:3] >= (self.psize_half + self.BBOX_SHIFT_MAX),
                sample_coord_all[:,:3] < (image_fused.shape[-3:] - self.psize_half
                - self.BBOX_SHIFT_MAX)), axis=1)]

            # Randomly sample APs.
            sample_coord_training = sample_coord_all[self.rng.choice(
                len(sample_coord_all), size=self.batch_size,
                replace=(self.batch_size > len(sample_coord_all)), shuffle=False)]

            # TODO: consider other distributions to boost performance on hardcase.
            removed_num = self.rng.integers(
                self.REMOVED_MAX[0], self.REMOVED_MAX[1], self.batch_size)

            # No CLS branches for now so all samples are positive.
            # Generate Positive training samples:
            # (1) Crop patches, (2) remove part of skel and seg voxels.
            for idx_patch in range(self.batch_size):
                # (1) Crop patches. AP (z,y,x) is the center of each patch.
                z,y,x = sample_coord_training[idx_patch, :3]
                # Augmentation: shift AP away from patchcenter for flexible learning.
                [z,y,x] = [z,y,x] + self.rng.integers(
                    -self.BBOX_SHIFT_MAX, self.BBOX_SHIFT_MAX+1 , 3)
                image_fused_patch_gt[idx_patch] = image_fused[
                    :,
                    z - self.psize_half[0] : z + self.psize_half[0],
                    y - self.psize_half[1] : y + self.psize_half[1],
                    x - self.psize_half[2] : x + self.psize_half[2]].copy()
                image_fused_patch[idx_patch] = image_fused_patch_gt[idx_patch].copy()            
                
                # (2) Remove part of skel, seg voxels, and edt.
                branch_coords = skeleton_skan.coordinates[
                    skeleton_skan.path(sample_coord_training[idx_patch, 3])]
                removed_idx = np.arange(
                    sample_coord_training[idx_patch, 4] - removed_num[idx_patch]//2,
                    sample_coord_training[idx_patch, 4] + (removed_num[idx_patch]+1)//2)
                # Augmentation: shift removed_idx for flexible learning. < self.IDX_TO_EP.
                removed_idx = np.clip(
                    removed_idx + self.rng.integers(-self.IDX_SHIFT, self.IDX_SHIFT+1),
                    a_min=7, a_max=(len(branch_coords) - 7))
                # Aug: multi-breaks via sparsely added points.
                kept_idx = self.rng.binomial(1, self.binomp, len(removed_idx))
                # Ensure that removed_idx is not null.
                if np.any(kept_idx>0):
                    removed_idx = removed_idx[kept_idx>0]
                # idx to coordinates. 
                removed_coord = branch_coords[removed_idx].astype(int)
                # The relative distance remains the same during conversion.
                removed_coord_patch = self.coordconvert_full2patch(removed_coord, [z,y,x])
                # (2-1) Remove skel and record edt for edt removal.
                edtlist_cur = []
                for coord_cur in removed_coord_patch:
                    if ((np.all(coord_cur) >= 0) and
                        (np.all(coord_cur < self.patch_size))):
                        # Make sure after shifting of seq_idx and bbox,
                        # removed_idx are still in the current bbox.
                        image_fused_patch[idx_patch, 2, coord_cur[0], coord_cur[1], coord_cur[2]] = 0
                        roi_mask_patch[idx_patch, 2, coord_cur[0], coord_cur[1], coord_cur[2]] = 1

                        if self.EDT_branch:
                            edtlist_cur.append(
                                image_fused_patch[idx_patch, 3, coord_cur[0],
                                coord_cur[1], coord_cur[2]])
                '''
                # (2-2) Remove voxels.
                # TODO: remove more on the bbox boundary.
                boxcoord_lower = removed_coord_patch.min(axis=0)
                boxcoord_upper = removed_coord_patch.max(axis=0)

                boxcoord_lower[boxcoord_lower < 0] = 0
                boxcoord_upper[boxcoord_upper > self.patch_size] = self.patch_size[
                    boxcoord_upper > self.patch_size]

                image_fused_patch[
                    idx_patch, 1,
                    boxcoord_lower[0]:boxcoord_upper[0],
                    boxcoord_lower[1]:boxcoord_upper[1],
                    boxcoord_lower[2]:boxcoord_upper[2]] = 0
                roi_mask_patch[
                    idx_patch, 1,
                    boxcoord_lower[0]:boxcoord_upper[0],
                    boxcoord_lower[1]:boxcoord_upper[1],
                    boxcoord_lower[2]:boxcoord_upper[2]] = 1
                '''

                if self.EDT_branch:
                    # (2-3) Remove edt.
                    # [Need finetune, + 2].
                    iternum = np.ceil(np.percentile(edtlist_cur, 80)).astype(np.int64) + self.EXTRA_DILATE_NUM
                    roi_mask_patch[idx_patch, 1] = roi_mask_patch[idx_patch, 2].copy()
                    roi_mask_patch[idx_patch, 1] = binary_dilation(
                        roi_mask_patch[idx_patch, 1],
                        self.dilate_struct,
                        iterations=iternum)

                    # Remove seg mask using the dilated RoI. 
                    image_fused_patch[idx_patch, 1] *= (1-roi_mask_patch[idx_patch, 1])
                    # Update EDT based on the removed seg mask.
                    # The EDT_GT is unchanged.
                    image_fused_patch[idx_patch, 3] = np.float32(
                        np.clip(distance_transform_edt(
                        image_fused_patch[idx_patch, 1]>0,
                        return_distances=True), a_min=0,
                        a_max=self.EDT_TH) / self.EDT_TH)

                # TODO: Remove part of image as training samples.

            '''
            # Generate Negative samples.
            # [Current] Negative samples only go through CLS branch.
            ## (1) Half Negative samples are positive samples without skel removal.
            ## (2) The other half Negative samples are randomly cropped patches
            ##     without EPs in central 6*6*6 region. [Need finetune].
            # (1) Extract the first half Negative samples (randomly from positive).
            idx_neg1 = self.rng.choice(self.batch_size_p,
                size=self.batch_size_p>>1, replace=False, shuffle=False)
            image_fused_patch_gt[self.batch_size_p:self.batch_size_3over4] = image_fused_patch_gt[idx_neg1].copy()
            # No need modifying on roi_mask_patch.
            image_fused_patch[self.batch_size_p:self.batch_size_3over4] = image_fused_patch_gt[idx_neg1].copy()
            
            # (2) Extract the other half Negative samples.
            z_list = self.rng.integers(self.patch_z, img_d - self.patch_z, 4*2048)
            y_list = self.rng.integers(self.patch_y, img_h - self.patch_y, 4*2048)
            x_list = self.rng.integers(self.patch_x, img_w - self.patch_x, 4*2048)
            coords_APs = np.array([z_list, y_list, x_list], dtype=np.int64).T

            branches_info_3D = branch_data_long[['skeleton-id', 'image-coord-src-0',
                    'image-coord-src-1', 'image-coord-src-2', 'skeleton-id', 'image-coord-dst-0',
                    'image-coord-dst-1', 'image-coord-dst-2']].values.reshape((-1, 4)).tolist()
            ep_list = [_ for _ in branches_info_3D if branches_info_3D.count(_)==1]
            ep_array = np.array(ep_list, dtype=np.int64) # (N, [skel_idx, z, y, x]).

            distmat_AP_EP = distance_matrix(coords_APs, ep_array[:,1:])
            np.fill_diagonal(distmat_AP_EP, self.DISTANCE_MAX)
            distmat_AP_EP_sorted = np.sort(distmat_AP_EP, axis=1)
            # Not too close, neither too far. [Need finetune].
            idx_valid = np.logical_and(distmat_AP_EP_sorted[:, 0] > 5, distmat_AP_EP_sorted[:, 1] <= 20)
            coords_APs = coords_APs[idx_valid]
            # Sample APs and corresponding RoIs.
            idx_neg2 = self.rng.choice(len(coords_APs),
                size=self.batch_size_p>>1, replace=(
                len(coords_APs) < self.batch_size_p>>1), shuffle=False)
            coords_APs = coords_APs[idx_neg2]
            for idx_patch in range(self.batch_size_p>>1):
                z,y,x = coords_APs[idx_patch]
                # Seems no need for shift aug because of random sampling already.
                # Start from batch_size_3over4.
                image_fused_patch_gt[idx_patch + self.batch_size_3over4] = image_fused[
                    :,
                    z - self.psize_half[0] : z + self.psize_half[0],
                    y - self.psize_half[1] : y + self.psize_half[1],
                    x - self.psize_half[2] : x + self.psize_half[2]].copy()
            # No need modifying on roi_mask_patch.
            image_fused_patch[(idx_patch+self.batch_size_3over4):] = image_fused_patch_gt[
                (idx_patch+self.batch_size_3over4):].copy()
            ''' 

            # Augmentation.
            # Rotation.
            if self.aug_flip:
                # Random flip on image patches.
                flip_idx = self.rng.integers(8)
                if flip_idx != 0:
                    image_fused_patch = np.flip(image_fused_patch, axis=self.flip_op[flip_idx])
                    roi_mask_patch = np.flip(roi_mask_patch, axis=self.flip_op[flip_idx])
                    image_fused_patch_gt = np.flip(image_fused_patch_gt, axis=self.flip_op[flip_idx])
        except:
            return None

        # Need shuffle samples in a batch or not?
        '''
        return (torch.from_numpy(image_fused_patch.copy()),
                torch.from_numpy(roi_mask_patch.copy()),
                torch.from_numpy(image_fused_patch_gt.copy()))
        '''
        return (image_fused_patch.copy(),
                roi_mask_patch.copy(),
                image_fused_patch_gt.copy())

    def is_connected(self, mask_pred_patch_sep, ep_pair_array_patch):
        predlabel_start = mask_pred_patch_sep[
            ep_pair_array_patch[0, 0],
            ep_pair_array_patch[0, 1],
            ep_pair_array_patch[0, 2]]
        predlabel_end = mask_pred_patch_sep[
            ep_pair_array_patch[1, 0],
            ep_pair_array_patch[1, 1],
            ep_pair_array_patch[1, 2]]
        
        return (predlabel_start) > 0 and (predlabel_start == predlabel_end)

    def is_trueconnection(self, label):
        cls_label = label != 0
        #cls_label = label > 0
        #cls_label = label > 1
        return cls_label

    def eval_report(self, acc_patch):
        
        TP, FN, FP, TN, FP_dual, All_count = acc_patch[:, 1:].sum(axis=0)

        accuracy = (TP + TN) / (All_count)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        precision_dual = TP / (TP + FP_dual)
        
        print('Accuracy: {:.02f}%; Recall: {:.02f}%; Precision: {:.02f}% ({:.02f}%).'.format(
            accuracy*100, recall*100, precision*100, precision_dual*100))
        
        return

    def getimagepatch(self, img, masks, BMAmask, patches_start_x,
            patches_start_z, syn_upper, syn_down, syn_y):
        '''
        Process an ODT volume and masks into input patches.
        '''
        patch_perimg = patches_start_x.shape[1]
        synpatches_height = syn_down - syn_upper
        # Placeholder.
        img_patch_perimg = np.zeros((patch_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=self.datatype)
        masks_patch_perimg = np.zeros((patch_perimg, 3, self.patch_z, self.patch_y, self.patch_x), dtype=np.bool8)
        BMAmask_patch_perimg = np.zeros((patch_perimg, self.patch_z, 1, self.patch_x), dtype=np.bool8)

        loss_mask_perimg = np.ones((patch_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=np.bool8)
        img_patch_gt_perimg = np.zeros((patch_perimg, self.patch_z, self.patch_y, self.patch_x), dtype=self.datatype)

        img_patch_exbot_perimg = np.zeros((patch_perimg, self.patch_z, synpatches_height, self.patch_x), dtype=self.datatype)
        masks_patch_exbot_perimg = np.zeros((patch_perimg, 3, self.patch_z, synpatches_height, self.patch_x), dtype=np.bool8)
        BMAmask_patch_exbot_perimg = np.zeros((patch_perimg, self.patch_z, 1, self.patch_x), dtype=np.bool8)

        # Extract patches.
        for i in range(patch_perimg):
            img_patch_perimg[i] = img[patches_start_z[0,i] : patches_start_z[0,i]+self.patch_z,
                                      :, patches_start_x[0,i] : patches_start_x[0,i]+self.patch_x].copy()
            masks_patch_perimg[i] = masks[:, patches_start_z[0,i] : patches_start_z[0,i]+self.patch_z,
                                          :, patches_start_x[0,i] : patches_start_x[0,i]+self.patch_x]
            BMAmask_patch_perimg[i,:,0,:] = BMAmask[patches_start_z[0,i] : patches_start_z[0,i]+self.patch_z,
                                                    patches_start_x[0,i] : patches_start_x[0,i]+self.patch_x]
             # Extract extra bottom-patches for synthesis.                                       
            img_patch_exbot_perimg[i] = img[patches_start_z[1,i] : patches_start_z[1,i]+self.patch_z,
                                            syn_upper:syn_down,
                                            patches_start_x[1,i] : patches_start_x[1,i]+self.patch_x]
            masks_patch_exbot_perimg[i] = masks[:, patches_start_z[1,i] : patches_start_z[1,i]+self.patch_z,
                                                syn_upper:syn_down,
                                                patches_start_x[1,i] : patches_start_x[1,i]+self.patch_x]
            BMAmask_patch_exbot_perimg[i,:,0,:] = BMAmask[patches_start_z[1,i] : patches_start_z[1,i]+self.patch_z,
                                                            patches_start_x[1,i] : patches_start_x[1,i]+self.patch_x]
            #print(i, img_patch_perimg.max(), masks_patch_perimg[i].sum((1,2,3)))

        if self.aug_intensity:
            # Intensity jittering augmentation.
            # All voxels jittering between [0.8, 1.2].
            ## For efficiency, only jittering on extracted patches, not full volume.
            jitter_factor = np.random.rand(
                patch_perimg, self.patch_z, self.patch_y, self.patch_x)*0.4 + 0.8
            img_patch_perimg *= jitter_factor

            # Prev: Syn-voxels suppression between [0.3, 0.7].
            # Syn-voxels suppression between [0.05, 0.45].
            supp_max = np.random.rand(
                patch_perimg, self.patch_z, synpatches_height, self.patch_x)*0.4 + 0.05
            # Although suppress all voxels but only FG' is used during syn.
            img_patch_exbot_perimg *= supp_max

        if self.aug_flip:
            # Random flip on image patches.
            flip_idx = np.random.randint(4)
            if flip_idx != 0:
                img_patch_perimg = np.flip(img_patch_perimg, axis=self.flip_op[flip_idx])
                masks_patch_perimg = np.flip(masks_patch_perimg, axis=self.flip_op_masks[flip_idx])
                BMAmask_patch_perimg = np.flip(BMAmask_patch_perimg, axis=self.flip_op[flip_idx])
            # Random flip on syn-image patches.
            flip_idx = np.random.randint(4)
            if flip_idx != 0:
                img_patch_exbot_perimg = np.flip(img_patch_exbot_perimg, axis=self.flip_op[flip_idx])
                masks_patch_exbot_perimg = np.flip(masks_patch_exbot_perimg, axis=self.flip_op_masks[flip_idx])
                BMAmask_patch_exbot_perimg = np.flip(BMAmask_patch_exbot_perimg, axis=self.flip_op[flip_idx])

        # Process the surface part (y-->700).
        # Input_B := I_B
        # GT_B := I_B*FG_B
        img_patch_gt_perimg[:,:,-self.patch_y_bottom:,:] = (img_patch_perimg[:,:,-self.patch_y_bottom:,:]
                                                            * masks_patch_perimg[:,0,:,-self.patch_y_bottom:,:])
        # Lossmask_B := ~MG_B*~BMA_B
        loss_mask_perimg[:,:,-self.patch_y_bottom:,:] = (~masks_patch_perimg[:,1,:,-self.patch_y_bottom:,:]
                                                         * ~BMAmask_patch_perimg)

        # Process the deep part (y-->0).
        # Input_T :=
        #   for syn-area: I'_B*FG'_B + I_T*~FG'_B
        #   for original: I_T
        synimg_FG = img_patch_exbot_perimg*masks_patch_exbot_perimg[:,0,...]
        img_patch_perimg[:,:, syn_y : syn_y+synpatches_height,:] *= ~masks_patch_exbot_perimg[:,0,...]
        img_patch_perimg[:,:, syn_y : syn_y+synpatches_height,:] += synimg_FG
        # GT_T := I'_B*FG'_B if synarea else 0 (assuming no signal here).
        img_patch_gt_perimg[:,:, syn_y : syn_y+synpatches_height,:] = synimg_FG.copy()
        # Lossmask_T [300,400) is 0. (Due to uncertainty, loss here is ignored.)
        loss_mask_perimg[:,:,self.patch_y_bottom:400,:] = 0
        # Lossmask_T for syn-area: ~BMA'_B (No MG'_B is injected so only ~BMA'_B.)
        loss_mask_perimg[:,:, syn_y : syn_y+synpatches_height,:] *= ~BMAmask_patch_exbot_perimg
        #print('--> ', img_patch_perimg.max(), loss_mask_perimg.sum(), img_patch_gt_perimg.max())

        # Augmentation to simulate broken vessels for all FG.
        # FG is mask.

        # Intensity suppresion 
        if self.aug_intensity:
            # FG mask of img_patch_perimg.
            img_FG_masks = np.zeros_like(img_patch_perimg, dtype=np.bool8)
            # Bottom FG := FG_B
            img_FG_masks[:,:,-self.patch_y_bottom:,:] = masks_patch_perimg[:,0,:,-self.patch_y_bottom:,:]
            # Top (syn. area) FG := FG'_B
            img_FG_masks[:,:,syn_y : syn_y+synpatches_height,:] = masks_patch_exbot_perimg[:,0,...]

            # Only suppress FG voxels.
            aug_factor = np.random.rand(
                patch_perimg, self.patch_z, self.patch_y, self.patch_x)
            img_patch_perimg = img_patch_perimg*(~img_FG_masks) + img_patch_perimg*img_FG_masks*aug_factor

        # May be ignored for efficiency.
        img_patch_perimg = np.clip(img_patch_perimg, a_min=0, a_max=1)
        img_patch_gt_perimg = np.clip(img_patch_gt_perimg, a_min=0, a_max=1)

        return img_patch_perimg, loss_mask_perimg, img_patch_gt_perimg


class MSD3D_VT_Dataset_Val(MSD3D_VT_Dataset):
    def __init__(self, config):
        super().__init__(config, training=False)
        assert self.training == False
        # Use fixed random generator to evaluate on the same set.
        self.rng = np.random.default_rng(self.eval_seed)
        # Use fixed aug in each eval so no need to turn off.
        #self.aug_intensity = False 
        #self.aug_flip = False

        self.datalist_val = []
        for img_path in tqdm(self.imglist_val):
            image_fused = np.load(img_path)
            skeleton_skan = skan_Skeleton(image_fused[2])
            branch_data = skan_summarize(skeleton_skan)
            # Normailization.
            image_fused = image_fused.astype(self.datatype)
            image_fused[0] = image_fused[0]/image_fused[0].max()

            self.datalist_val.append([
                image_fused,
                skeleton_skan,
                branch_data])

        self.datalist = self.datalist_val

    def on_epoch_end(self):
        # Reset generator seed.
        self.rng = np.random.default_rng(self.eval_seed)
    
    def __len__(self):
        return len(self.datalist_val)


if __name__ == "__main__":
    pass