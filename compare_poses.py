import argparse
import numpy as np
from pipelime.sequences.readers.filesystem import UnderfolderReader

from utils.vis_utils import show_poses


def compare_poses(in_folder : str, pred_key : str, gt_key : str):
    
    uf = UnderfolderReader(in_folder)

  
    # read data
    pred_poses = []
    gt_poses = []
    for sample in uf:
        
       
        pred_pose = np.array(sample[pred_key], dtype=np.float32)
        gt_pose = np.array(sample[gt_key], dtype=np.float32)

        pred_poses.append(pred_pose)
        gt_poses.append(gt_pose)

    pred = np.stack(pred_poses, axis=0)  # [N_IMAGES, 4, 4]
    gt = np.stack(gt_poses, axis=0)      # [N_IMAGES, 4, 4]

    #traslation error 

    t_errors = np.linalg.norm((pred-gt)[:,:3,3],axis=1) 

    #rotation z-error 

    z_pred = pred[:,:3,2]
    z_gt = gt[:,:3,2]

    r_errors = np.einsum("ij,ij->i", z_pred, z_gt)

    #rotation error 

    gt_rot = gt[:,:3,:3]
    pred_rot = pred[:,:3,:3]

    gt_rot_T = np.transpose(gt_rot,axes=[0,2,1])

    diff_matrix = np.matmul(pred_rot,gt_rot_T)

    theta = np.arccos(np.clip((np.trace(diff_matrix,axis1=1,axis2=2)-1)/2, -1, 1))

    theta_angles = np.rad2deg(theta)

    print(f'translation errors : mean {np.mean(t_errors)}, std : {np.std(t_errors)}, median: {np.median(t_errors)} min : {np.min(t_errors)}, max : {np.max(t_errors)}')
    print(f'rotation z errors : mean {np.mean(r_errors)}, std : {np.std(r_errors)}, median: {np.median(t_errors)}, min : {np.min(r_errors)}, max : {np.max(r_errors)}')
    print(f'rotation errors : mean {np.mean(theta_angles)}, std : {np.std(theta_angles)}, median: {np.median(t_errors)}, min : {np.min(theta_angles)}, max : {np.max(theta_angles)}')

    print(f'pose with max translation error {np.argmax(t_errors)}')

    show_poses(in_folder,pred_key,t_errors)

    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input', dest='in_folder', type=str, help='input folder where poses are stored', required=True)
    parser.add_argument('--pred_key', dest='pred_key', type=str, help='uf key for pred poses', required=True)
    parser.add_argument('--gt_key', dest='gt_key', type=str, help='uf key for gt poses', required=True)

    args = parser.parse_args()
    
    compare_poses(args.in_folder, args.pred_key, args.gt_key)



    