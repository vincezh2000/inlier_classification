"""
General utility functions
Author: Shengyu Huang
Last modified: 30.11.2020
"""

import glob
import math
import os
import pickle
import random
import re
from collections import defaultdict

import nibabel.quaternions as nq
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import yaml
import copy

_EPS = 1e-7  # To prevent division by zero

def get_rot_trans(H):

    R = torch.zeros((H.shape[0], 3, 3))
    t = torch.zeros((H.shape[0], 3, 1))

    R = H[:, 0:3, 0:3]
    t = H[:, 0:3, 3]

    return R, t


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.pipelines.visualization.draw_geometries(
        [source_temp, target_temp],
        zoom=0.4459,
        front=[0.9288, -0.2951, -0.2242],
        lookat=[1.6784, 2.0612, 1.4451],
        up=[-0.3402, -0.9189, -0.1996],
    )
    
class Logger:
    def __init__(self, path):
        self.path = path
        self.fw = open(self.path + "/log", "a")

    def write(self, text):
        self.fw.write(text)
        self.fw.flush()

    def close(self):
        self.fw.close()


def save_obj(obj, path):
    """
    save a dictionary to a pickle file
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def square_distance(src, dst, normalised=False):
    """
    Calculate Euclid distance between each two points.
    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    if normalised:
        dist += 2
    else:
        dist += torch.sum(src**2, dim=-1)[:, :, None]
        dist += torch.sum(dst**2, dim=-1)[:, None, :]

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist


def validate_gradient(model):
    """
    Confirm all the gradients are non-nan and non-inf
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.any(torch.isnan(param.grad)):
                return False
            if torch.any(torch.isinf(param.grad)):
                return False
    return True


def natural_key(string_):
    """
    Sort strings by numbers in the name
    """
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_)]


def rotation_error(R1, R2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix.
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args:
        R1 (torch tensor): Estimated rotation matrices [b,3,3]
        R2 (torch tensor): Ground truth rotation matrices [b,3,3]
    Returns:
        ae (torch tensor): Rotation error in angular degreees [b,1]
    """
    R_ = torch.matmul(R1.transpose(1, 2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)
    pi = torch.Tensor([math.pi])
    ae = 180.0 * ae / pi.to(ae.device).type(ae.dtype)

    return ae


def translation_error(t1, t2):
    """
    Torch batch implementation of the rotation error between the estimated and the ground truth rotatiom matrix.
    Rotation error is defined as r_e = \arccos(\frac{Trace(\mathbf{R}_{ij}^{T}\mathbf{R}_{ij}^{\mathrm{GT}) - 1}{2})
    Args:
        t1 (torch tensor): Estimated translation vectors [b,3,1]
        t2 (torch tensor): Ground truth translation vectors [b,3,1]
    Returns:
        te (torch tensor): translation error in meters [b,1]
    """
    return torch.norm(t1 - t2, dim=(1, 2))


def computeTransformationErr(trans, info):
    """
    Computer the transformation error as an approximation of the RMSE of corresponding points.
    More informaiton at http://redwood-data.org/indoor/registration.html

    Args:
    trans (numpy array): transformation matrices [n,4,4]
    info (numpy array): covariance matrices of the gt transformation paramaters [n,4,4]
    Returns:
    p (float): transformation error
    """

    t = trans[:3, 3]
    r = trans[:3, :3]
    q = nq.mat2quat(r)
    er = np.concatenate([t, q[1:]], axis=0)
    p = er.reshape(1, 6) @ info @ er.reshape(6, 1) / info[0, 0]

    return p.item()


def read_trajectory(filename, dim=4):
    """
    Function that reads a trajectory saved in the 3DMatch/Redwood format to a numpy array.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    Returns:
    final_keys (dict): indices of pairs with more than 30% overlap (only this ones are included in the gt file)
    traj (numpy array): gt pairwise transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as f:
        lines = f.readlines()

        # Extract the point cloud pairs
        keys = lines[0 :: (dim + 1)]
        temp_keys = []
        for i in range(len(keys)):
            temp_keys.append(keys[i].split("\t")[0:3])

        final_keys = []
        for i in range(len(temp_keys)):
            final_keys.append([temp_keys[i][0].strip(), temp_keys[i][1].strip(), temp_keys[i][2].strip()])

        traj = []
        for i in range(len(lines)):
            if i % 5 != 0:
                traj.append(lines[i].split("\t")[0:dim])

        traj = np.asarray(traj, dtype=np.float).reshape(-1, dim, dim)

        final_keys = np.asarray(final_keys)

        return final_keys, traj


def read_trajectory_info(filename, dim=6):
    """
    Function that reads the trajectory information saved in the 3DMatch/Redwood format to a numpy array.
    Information file contains the variance-covariance matrix of the transformation paramaters.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html

    Args:
    filename (str): path to the '.txt' file containing the trajectory information data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    Returns:
    n_frame (int): number of fragments in the scene
    cov_matrix (numpy array): covariance matrix of the transformation matrices for n pairs[n,dim, dim]
    """

    with open(filename) as fid:
        contents = fid.readlines()
    n_pairs = len(contents) // 7
    assert len(contents) == 7 * n_pairs
    info_list = []
    n_frame = 0

    for i in range(n_pairs):
        frame_idx0, frame_idx1, n_frame = [int(item) for item in contents[i * 7].strip().split()]
        info_matrix = np.concatenate(
            [np.fromstring(item, sep="\t").reshape(1, -1) for item in contents[i * 7 + 1 : i * 7 + 7]], axis=0
        )
        info_list.append(info_matrix)

    cov_matrix = np.asarray(info_list, dtype=np.float).reshape(-1, dim, dim)

    return n_frame, cov_matrix


def extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj):
    """
    Extract only those transformation matrices from the ground truth trajectory that are also in the estimated trajectory.

    Args:
    est_pairs (numpy array): indices of point cloud pairs with enough estimated overlap [m, 3]
    gt_pairs (numpy array): indices of gt overlaping point cloud pairs [n,3]
    gt_traj (numpy array): 3d array of the gt transformation parameters [n,4,4]
    Returns:
    ext_traj (numpy array): gt transformation parameters for the point cloud pairs from est_pairs [m,4,4]
    """
    ext_traj = np.zeros((len(est_pairs), 4, 4))

    for est_idx, pair in enumerate(est_pairs):
        pair[2] = gt_pairs[0][2]
        gt_idx = np.where((gt_pairs == pair).all(axis=1))[0]

        ext_traj[est_idx, :, :] = gt_traj[gt_idx, :, :]

    return ext_traj


def write_trajectory(traj, metadata, filename, dim=4):
    """
    Writes the trajectory into a '.txt' file in 3DMatch/Redwood format.
    Format specification can be found at http://redwood-data.org/indoor/fileformat.html
    Args:
    traj (numpy array): trajectory for n pairs[n,dim, dim]
    metadata (numpy array): file containing metadata about fragment numbers [n,3]
    filename (str): path where to save the '.txt' file containing trajectory data
    dim (int): dimension of the transformation matrix (4x4 for 3D data)
    """

    with open(filename, "w") as f:
        for idx in range(traj.shape[0]):
            # Only save the transfromation parameters for which the overlap threshold was satisfied
            if metadata[idx][2]:
                p = traj[idx, :, :].tolist()
                f.write("\t".join(map(str, metadata[idx])) + "\n")
                f.write("\n".join("\t".join(map("{0:.12f}".format, p[i])) for i in range(dim)))
                f.write("\n")


def read_pairs(src_path, tgt_path, n_points):
    # get pointcloud
    src = torch.load(src_path)
    tgt = torch.load(tgt_path)
    src_pcd, src_embedding = src["coords"], src["feats"]
    tgt_pcd, tgt_embedding = tgt["coords"], tgt["feats"]

    # permute and randomly select 2048/1024 points
    if src_pcd.shape[0] > n_points:
        src_permute = np.random.permutation(src_pcd.shape[0])[:n_points]
    else:
        src_permute = np.random.choice(src_pcd.shape[0], n_points)
    if tgt_pcd.shape[0] > n_points:
        tgt_permute = np.random.permutation(tgt_pcd.shape[0])[:n_points]
    else:
        tgt_permute = np.random.choice(tgt_pcd.shape[0], n_points)

    src_pcd, src_embedding = src_pcd[src_permute], src_embedding[src_permute]
    tgt_pcd, tgt_embedding = tgt_pcd[tgt_permute], tgt_embedding[tgt_permute]
    return src_pcd, src_embedding, tgt_pcd, tgt_embedding


def evaluate_registration(num_fragment, result, result_pairs, gt_pairs, gt, gt_info, err2=0.2):
    """
    Evaluates the performance of the registration algorithm according to the evaluation protocol defined
    by the 3DMatch/Redwood datasets. The evaluation protocol can be found at http://redwood-data.org/indoor/registration.html

    Args:
    num_fragment (int): path to the '.txt' file containing the trajectory information data
    result (numpy array): estimated transformation matrices [n,4,4]
    result_pairs (numpy array): indices of the point cloud for which the transformation matrix was estimated (m,3)
    gt_pairs (numpy array): indices of the ground truth overlapping point cloud pairs (n,3)
    gt (numpy array): ground truth transformation matrices [n,4,4]
    gt_cov (numpy array): covariance matrix of the ground truth transfromation parameters [n,6,6]
    err2 (float): threshold for the RMSE of the gt correspondences (default: 0.2m)
    Returns:
    precision (float): mean registration precision over the scene (not so important because it can be increased see papers)
    recall (float): mean registration recall over the scene (deciding parameter for the performance of the algorithm)
    """

    err2 = err2**2
    gt_mask = np.zeros((num_fragment, num_fragment), dtype=np.int)
    flags = []

    for idx in range(gt_pairs.shape[0]):
        i = int(gt_pairs[idx, 0])
        j = int(gt_pairs[idx, 1])

        # Only non consecutive pairs are tested
        if j - i > 1:
            gt_mask[i, j] = idx

    n_gt = np.sum(gt_mask > 0)

    good = 0
    n_res = 0
    for idx in range(result_pairs.shape[0]):
        i = int(result_pairs[idx, 0])
        j = int(result_pairs[idx, 1])
        pose = result[idx, :, :]

        if gt_mask[i, j] > 0:
            n_res += 1
            gt_idx = gt_mask[i, j]
            p = computeTransformationErr(np.linalg.inv(gt[gt_idx, :, :]) @ pose, gt_info[gt_idx, :, :])
            if p <= err2:
                good += 1
                flags.append(0)
            else:
                flags.append(1)
        else:
            flags.append(2)
    if n_res == 0:
        n_res += 1e6
    precision = good * 1.0 / n_res
    recall = good * 1.0 / n_gt

    return precision, recall, flags


def benchmark(est_folder, gt_folder):
    scenes = sorted(os.listdir(gt_folder))
    scene_names = [os.path.join(gt_folder, ele) for ele in scenes]

    re_per_scene = defaultdict(list)
    te_per_scene = defaultdict(list)
    re_all, te_all, precision, recall = [], [], [], []
    n_valids = []

    short_names = ["Kitchen", "Home 1", "Home 2", "Hotel 1", "Hotel 2", "Hotel 3", "Study", "MIT Lab"]
    with open(f"{est_folder}/result", "w") as f:
        f.write(("Scene\t¦ prec.\t¦ rec.\t¦ re\t¦ te\t¦ samples\t¦\n"))

        for idx, scene in enumerate(scene_names):
            # ground truth info
            gt_pairs, gt_traj = read_trajectory(os.path.join(scene, "gt.log"))
            n_valid = 0
            for ele in gt_pairs:
                diff = abs(int(ele[0]) - int(ele[1]))
                n_valid += diff > 1
            n_valids.append(n_valid)

            n_fragments, gt_traj_cov = read_trajectory_info(os.path.join(scene, "gt.info"))

            # estimated info
            est_pairs, est_traj = read_trajectory(os.path.join(est_folder, scenes[idx], "est.log"))

            temp_precision, temp_recall, c_flag = evaluate_registration(
                n_fragments, est_traj, est_pairs, gt_pairs, gt_traj, gt_traj_cov
            )

            # Filter out the estimated rotation matrices
            ext_gt_traj = extract_corresponding_trajectors(est_pairs, gt_pairs, gt_traj)

            re = (
                rotation_error(torch.from_numpy(ext_gt_traj[:, 0:3, 0:3]), torch.from_numpy(est_traj[:, 0:3, 0:3]))
                .cpu()
                .numpy()[np.array(c_flag) == 0]
            )
            te = (
                translation_error(torch.from_numpy(ext_gt_traj[:, 0:3, 3:4]), torch.from_numpy(est_traj[:, 0:3, 3:4]))
                .cpu()
                .numpy()[np.array(c_flag) == 0]
            )

            re_per_scene["mean"].append(np.mean(re))
            re_per_scene["median"].append(np.median(re))
            re_per_scene["min"].append(np.min(re))
            re_per_scene["max"].append(np.max(re))

            te_per_scene["mean"].append(np.mean(te))
            te_per_scene["median"].append(np.median(te))
            te_per_scene["min"].append(np.min(te))
            te_per_scene["max"].append(np.max(te))

            re_all.extend(re.reshape(-1).tolist())
            te_all.extend(te.reshape(-1).tolist())

            precision.append(temp_precision)
            recall.append(temp_recall)

            f.write(
                "{}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:.3f}\t¦ {:3d}¦\n".format(
                    short_names[idx], temp_precision, temp_recall, np.median(re), np.median(te), n_valid
                )
            )
            np.save(f"{est_folder}/{scenes[idx]}/flag.npy", c_flag)

        weighted_precision = (np.array(n_valids) * np.array(precision)).sum() / np.sum(n_valids)

        f.write("Mean precision: {:.3f}: +- {:.3f}\n".format(np.mean(precision), np.std(precision)))
        f.write("Weighted precision: {:.3f}\n".format(weighted_precision))

        f.write(
            "Mean median RRE: {:.3f}: +- {:.3f}\n".format(
                np.mean(re_per_scene["median"]), np.std(re_per_scene["median"])
            )
        )
        f.write(
            "Mean median RTE: {:.3F}: +- {:.3f}\n".format(
                np.mean(te_per_scene["median"]), np.std(te_per_scene["median"])
            )
        )
    f.close()


def fmr_wrt_distance(data, split, inlier_ratio_threshold=0.05):
    """
    calculate feature match recall wrt distance threshold
    """
    fmr_wrt_distance = []
    for distance_threshold in range(1, 21):
        inlier_ratios = []
        distance_threshold /= 100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)
        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0] : ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_distance.append(fmr * 100)
    return fmr_wrt_distance


def fmr_wrt_inlier_ratio(data, split, distance_threshold=0.1):
    """
    calculate feature match recall wrt inlier ratio threshold
    """
    fmr_wrt_inlier = []
    for inlier_ratio_threshold in range(1, 21):
        inlier_ratios = []
        inlier_ratio_threshold /= 100.0
        for idx in range(data.shape[0]):
            inlier_ratio = (data[idx] < distance_threshold).mean()
            inlier_ratios.append(inlier_ratio)

        fmr = 0
        for ele in split:
            fmr += (np.array(inlier_ratios[ele[0] : ele[1]]) > inlier_ratio_threshold).mean()
        fmr /= 8
        fmr_wrt_inlier.append(fmr * 100)

    return fmr_wrt_inlier


def write_est_trajectory(gt_folder, exp_dir, tsfm_est):
    """
    Write the estimated trajectories
    """
    scene_names = sorted(os.listdir(gt_folder))
    count = 0
    for scene_name in scene_names:
        gt_pairs, gt_traj = read_trajectory(os.path.join(gt_folder, scene_name, "gt.log"))
        est_traj = []
        for i in range(len(gt_pairs)):
            est_traj.append(tsfm_est[count])
            count += 1

        # write the trajectory
        c_directory = os.path.join(exp_dir, scene_name)
        os.makedirs(c_directory)
        write_trajectory(np.array(est_traj), gt_pairs, os.path.join(c_directory, "est.log"))


def to_tensor(array):
    """
    Convert array to tensor
    """
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device("cpu"):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_tsfm(rot, trans):
    tsfm = np.eye(4)
    tsfm[:3, :3] = rot
    tsfm[:3, 3] = trans.flatten()
    return tsfm


def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def get_correspondences(src_pcd, tgt_pcd, trans, search_voxel_size, K=None):
    src_pcd.transform(trans)
    pcd_tree = o3d.geometry.KDTreeFlann(tgt_pcd)

    correspondences = []
    for i, point in enumerate(src_pcd.points):
        [count, idx, _] = pcd_tree.search_radius_vector_3d(point, search_voxel_size)
        if K is not None:
            idx = idx[:K]
        for j in idx:
            correspondences.append([i, j])

    correspondences = np.array(correspondences)
    correspondences = torch.from_numpy(correspondences)
    return correspondences


def get_blue():
    """
    Get color blue for rendering
    """
    return [0, 0.651, 0.929]


def get_yellow():
    """
    Get color yellow for rendering
    """
    return [1, 0.706, 0]


def random_sample(pcd, feats, N):
    """
    Do random sampling to get exact N points and associated features
    pcd:    [N,3]
    feats:  [N,C]
    """
    if isinstance(pcd, torch.Tensor):
        n1 = pcd.size(0)
    elif isinstance(pcd, np.ndarray):
        n1 = pcd.shape[0]

    if n1 == N:
        return pcd, feats

    if n1 > N:
        choice = np.random.permutation(n1)[:N]
    else:
        choice = np.random.choice(n1, N)

    return pcd[choice], feats[choice]


def get_angle_deviation(R_pred, R_gt):
    """
    Calculate the angle deviation between two rotaion matrice
    The rotation error is between [0,180]
    Input:
        R_pred: [B,3,3]
        R_gt  : [B,3,3]
    Return:
        degs:   [B]
    """
    R = np.matmul(R_pred, R_gt.transpose(0, 2, 1))
    tr = np.trace(R, 0, 1, 2)
    rads = np.arccos(np.clip((tr - 1) / 2, -1, 1))  # clip to valid range
    degs = rads / np.pi * 180

    return degs


def ransac_pose_estimation(src_pcd, tgt_pcd, src_feat, tgt_feat, mutual=False, distance_threshold=0.05, ransac_n=3):
    """
    RANSAC pose estimation with two checkers
    We follow D3Feat to set ransac_n = 3 for 3DMatch and ransac_n = 4 for KITTI.
    For 3DMatch dataset, we observe significant improvement after changing ransac_n from 4 to 3.
    """
    if mutual:
        if torch.cuda.device_count() >= 1:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        src_feat, tgt_feat = to_tensor(src_feat), to_tensor(tgt_feat)
        scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0, 1).to(device)).cpu()
        selection = mutual_selection(scores[None, :, :])[0]
        row_sel, col_sel = np.where(selection)
        corrs = o3d.utility.Vector2iVector(np.array([row_sel, col_sel]).T)
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        result_ransac = o3d.registration.registration_ransac_based_on_correspondence(
            source=src_pcd,
            target=tgt_pcd,
            corres=corrs,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n=4,
            criteria=o3d.registration.RANSACConvergenceCriteria(50000, 1000),
        )
    else:
        src_pcd = to_o3d_pcd(src_pcd)
        tgt_pcd = to_o3d_pcd(tgt_pcd)
        src_feats = to_o3d_feats(src_feat)
        tgt_feats = to_o3d_feats(tgt_feat)

        result_ransac = o3d.registration.registration_ransac_based_on_feature_matching(
            src_pcd,
            tgt_pcd,
            src_feats,
            tgt_feats,
            distance_threshold,
            o3d.registration.TransformationEstimationPointToPoint(False),
            ransac_n,
            [
                o3d.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            o3d.registration.RANSACConvergenceCriteria(50000, 1000),
        )

    return result_ransac.transformation


def get_inlier_ratio(src_pcd, tgt_pcd, src_feat, tgt_feat, rot, trans, inlier_distance_threshold=0.1):
    """
    Compute inlier ratios with and without mutual check, return both
    """
    src_pcd = to_tensor(src_pcd)
    tgt_pcd = to_tensor(tgt_pcd)
    src_feat = to_tensor(src_feat)
    tgt_feat = to_tensor(tgt_feat)
    rot, trans = to_tensor(rot), to_tensor(trans)

    results = dict()
    results["w"] = dict()
    results["wo"] = dict()

    if torch.cuda.device_count() >= 1:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    src_pcd = (torch.matmul(rot, src_pcd.transpose(0, 1)) + trans).transpose(0, 1)
    scores = torch.matmul(src_feat.to(device), tgt_feat.transpose(0, 1).to(device)).cpu()

    ########################################
    # 1. calculate inlier ratios wo mutual check
    _, idx = scores.max(-1)
    dist = torch.norm(src_pcd - tgt_pcd[idx], dim=1)
    results["wo"]["distance"] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results["wo"]["inlier_ratio"] = c_inlier_ratio

    ########################################
    # 2. calculate inlier ratios w mutual check
    selection = mutual_selection(scores[None, :, :])[0]
    row_sel, col_sel = np.where(selection)
    dist = torch.norm(src_pcd[row_sel] - tgt_pcd[col_sel], dim=1)
    results["w"]["distance"] = dist.numpy()

    c_inlier_ratio = (dist < inlier_distance_threshold).float().mean()
    results["w"]["inlier_ratio"] = c_inlier_ratio

    return results


def mutual_selection(score_mat):
    """
    Return a {0,1} matrix, the element is 1 if and only if it's maximum along both row and column

    Args: np.array()
        score_mat:  [B,N,N]
    Return:
        mutuals:    [B,N,N]
    """
    score_mat = to_array(score_mat)
    if score_mat.ndim == 2:
        score_mat = score_mat[None, :, :]

    mutuals = np.zeros_like(score_mat)
    for i in range(score_mat.shape[0]):  # loop through the batch
        c_mat = score_mat[i]
        flag_row = np.zeros_like(c_mat)
        flag_column = np.zeros_like(c_mat)

        max_along_row = np.argmax(c_mat, 1)[:, None]
        max_along_column = np.argmax(c_mat, 0)[None, :]
        np.put_along_axis(flag_row, max_along_row, 1, 1)
        np.put_along_axis(flag_column, max_along_column, 1, 0)
        mutuals[i] = (flag_row.astype(np.bool)) & (flag_column.astype(np.bool))
    return mutuals.astype(np.bool)


def get_scene_split(whichbenchmark):
    """
    Just to check how many valid fragments each scene has
    """
    assert whichbenchmark in ["3DMatch", "3DLoMatch"]
    folder = f"configs/benchmarks/{whichbenchmark}/*/gt.log"

    scene_files = sorted(glob.glob(folder))
    split = []
    count = 0
    for eachfile in scene_files:
        gt_pairs, gt_traj = read_trajectory(eachfile)
        split.append([count, count + len(gt_pairs)])
        count += len(gt_pairs)
    return split
