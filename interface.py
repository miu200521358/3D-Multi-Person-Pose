from lib.models import networkgcn, networktcn
import torch
import numpy as np
from TorchSUL import Model as M
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from glob import glob
import os
from collections import defaultdict
import scipy.io as sio
import json


# https://github.com/Fang-Haoshu/Halpe-FullBody#keypoints-format
HALPE_KEYPOINTS = {
    "Nose": 0,
    "LEye": 1,
    "REye": 2,
    "LEar": 3,
    "REar": 4,
    "LShoulder": 5,
    "RShoulder": 6,
    "LElbow": 7,
    "RElbow": 8,
    "LWrist": 9,
    "RWrist": 10,
    "LHip": 11,
    "RHip": 12,
    "LKnee": 13,
    "RKnee": 14,
    "LAnkle": 15,
    "RAnkle": 16,
    "Head": 17,
    "Neck": 18,
    "Pelvis": 19,
    "LBigToe": 20,
    "RBigToe": 21,
    "LSmallToe": 22,
    "RSmallToe": 23,
    "LHeel": 24,
    "RHeel": 25,
    "Spine": 26,  # 計算して追加
}

H36M_KEYPOINTS = {
    "Pelvis": 0,
    "LHip": 1,
    "LKnee": 2,
    "LAnkle": 3,
    "RHip": 4,
    "RKnee": 5,
    "RAnkle": 6,
    "Spine": 7,
    "Neck": 8,
    "Nose": 9,
    "Head": 10,
    "LShoulder": 11,
    "LElbow": 12,
    "LWrist": 13,
    "RShoulder": 14,
    "RElbow": 15,
    "RWrist": 16,
}

H36M_BONE_PAIRS = np.array(
    [
        np.array([8, 9]),
        np.array([9, 10]),
        np.array([8, 14]),
        np.array([14, 15]),
        np.array([15, 16]),
        np.array([8, 11]),
        np.array([12, 13]),
        np.array([11, 12]),
        np.array([8, 7]),
        np.array([7, 0]),
        np.array([4, 5]),
        np.array([5, 6]),
        np.array([0, 4]),
        np.array([0, 1]),
        np.array([1, 2]),
        np.array([2, 3]),
    ]
)

BASE_PATH = "E:\\MMD\\MikuMikuDance_v926x64\\Work\\201805_auto\\01\\heart\\heart_0-650_mp4_20221009_183826\\02_alphapose"


def calculate_mupots_topdown_pts():
    bone_matrix = np.zeros([16, 17], dtype=np.float32)
    for i, pair in enumerate(H36M_BONE_PAIRS):
        bone_matrix[i, pair[0]] = -1
        bone_matrix[i, pair[1]] = 1
    bone_matrix_inv = np.linalg.pinv(bone_matrix)
    bone_matrix_inv = torch.from_numpy(bone_matrix_inv)
    bone_matrix = torch.from_numpy(bone_matrix)

    seq_len = 243
    netgcn = networkgcn.TransNet(256, 17)
    nettcn = networktcn.Refine2dNet(17, seq_len)

    # initialize the network with dumb input
    x_dumb = torch.zeros(2, 17, 2)
    affb = torch.ones(2, 16, 16) / 16
    affpts = torch.ones(2, 17, 17) / 17
    netgcn(x_dumb, affpts, affb, bone_matrix, bone_matrix_inv)
    x_dumb = torch.zeros(2, 243, 17 * 3)
    nettcn(x_dumb)

    # load networks
    M.Saver(netgcn).restore("./ckpts/model_gcnwild/")
    M.Saver(nettcn).restore("./ckpts/model_tcn/")

    # push to gpu
    netgcn.cuda()
    netgcn.eval()
    nettcn.cuda()
    nettcn.eval()
    bone_matrix = bone_matrix.cuda()
    bone_matrix_inv = bone_matrix_inv.cuda()

    all_fnos = {}
    all_p2d = {}
    all_affpts = {}
    all_affb = {}
    all_fnos = {}
    results = {}
    for persion_json_path in glob(
        os.path.join(
            BASE_PATH,
            "*.json",
        )
    ):
        if "alphapose-results.json" in persion_json_path:
            continue

        # キーフレ番号
        fnos = []
        # 関節の位置
        fp2ds = []
        # 関節の信頼度
        faffpts = []

        json_datas = {}
        with open(persion_json_path, "r") as f:
            json_datas = json.load(f)

        # 人物INDEX
        pname, _ = os.path.splitext(os.path.basename(persion_json_path))

        for fno in tqdm(
            sorted([int(f) for f in json_datas["estimation"].keys()]),
            desc=f"No.{pname} ... ",
        ):
            frame_json_data = json_datas["estimation"][str(fno)]
            fno = int(fno)

            fnos.append(fno)

            kps = np.array(frame_json_data["2d-keypoints"]).reshape(-1, 3)
            spine = np.mean([kps[19], kps[18]], axis=0)
            kps2 = np.vstack([kps, spine])
            # HALPE -> H36M
            kps3 = kps2[
                (19, 12, 14, 16, 11, 13, 15, 26, 18, 0, 17, 5, 7, 9, 6, 8, 10), :
            ]

            fp2ds.append(kps3[:, :2])
            faffpts.append(kps3[:, 2])

        p2d = np.array(fp2ds, dtype=np.float32)
        faffpts = np.array(faffpts, dtype=np.float32)

        affpts = np.tile(faffpts, 17).reshape(faffpts.shape[0], 17, 17)

        affpt_a = faffpts[:, H36M_BONE_PAIRS[:, 0]]
        affpt_b = faffpts[:, H36M_BONE_PAIRS[:, 1]]

        affpt_a_tile = np.tile(affpt_a, 16).reshape(affpt_a.shape[0], 16, 16)
        affpt_b_tile = np.tile(affpt_b, 16).reshape(affpt_b.shape[0], 16, 16)

        affb = np.mean([affpt_a_tile, affpt_b_tile], axis=0)

        all_p2d[pname] = p2d
        all_affpts[pname] = affpts
        all_affb[pname] = affb

        p2d = torch.from_numpy(p2d).cuda() / 1024
        scale = p2d[:, 8:9, 1:2] - p2d[:, 0:1, 1:2]
        p2d = p2d / scale
        p2d = p2d - p2d[:, 0:1]
        affb = torch.from_numpy(affb).cuda()
        affpts = torch.from_numpy(affpts).cuda()
        with torch.no_grad():
            pred = netgcn(p2d, affpts, affb, bone_matrix, bone_matrix_inv)
            pred = pred.unsqueeze(0).unsqueeze(0)
            pred = pred - pred[:, :, :, :1]
            # pred = pred * occmask
            pred = F.pad(
                pred, (0, 0, 0, 0, seq_len // 2, seq_len // 2), mode="replicate"
            )
            pred = pred.squeeze()
            pred = nettcn.evaluate(pred)

            # pickle.dump(pred.cpu().numpy(), open(ptsfile.replace('p2ds/', 'pred/'), 'wb'))
            pred = pred.cpu().numpy()
            results[pname] = pred
        all_fnos[pname] = fnos

    return results, all_fnos, all_p2d, all_affpts, all_affb


def calculate_mupots_topdown_depth(td_pts_results, all_p2d, all_affpts, all_affb):
    seq_len = 243
    nettcn = networktcn.Refine2dNet(
        17, seq_len, input_dimension=2, output_dimension=1, output_pts=1
    )
    x_dumb = torch.zeros(2, 243, 17 * 2)
    nettcn(x_dumb)
    M.Saver(nettcn).restore("./ckpts/model_root/")
    nettcn.cuda()
    nettcn.eval()

    results = {}
    gts = {}
    for pname, td_pts_result in td_pts_results.items():
        p2d = all_p2d[pname]
        affpts = all_affpts[pname]
        affb = all_affb[pname]

        p2d = torch.from_numpy(p2d).cuda() / 915

        with torch.no_grad():
            p2d = p2d.unsqueeze(0).unsqueeze(0)
            p2d = F.pad(p2d, (0, 0, 0, 0, seq_len // 2, seq_len // 2), mode="replicate")
            p2d = p2d.squeeze()
            pred = nettcn.evaluate(p2d)
            pred = pred.cpu().numpy()

        # do pa alignment
        results[pname] = pred

    return results


def exec():

    (
        td_pts_results,
        all_fnos,
        all_p2d,
        all_affpts,
        all_affb,
    ) = calculate_mupots_topdown_pts()

    td_depth_results = calculate_mupots_topdown_depth(
        td_pts_results, all_p2d, all_affpts, all_affb
    )

    for persion_json_path in glob(
        os.path.join(
            BASE_PATH,
            "*.json",
        )
    ):
        if "alphapose-results.json" in persion_json_path:
            continue

        json_datas = {}
        with open(persion_json_path, "r") as f:
            json_datas = json.load(f)

        # 人物INDEX
        pname, _ = os.path.splitext(os.path.basename(persion_json_path))

        for fidx in tqdm(
            range(len(json_datas["estimation"].keys())),
            desc=f"No.{pname} ... ",
        ):
            fno = str(all_fnos[pname][fidx])

            json_datas["estimation"][fno]["depth"] = (
                float(td_depth_results[pname][fidx]) * 100
            )

            json_datas["estimation"][fno]["mpp_joints"] = {}

            for jname, jidx in H36M_KEYPOINTS.items():
                json_datas["estimation"][fno]["mpp_joints"][jname] = {
                    "x": float(td_pts_results[pname][fidx][jidx][0]) * 100,
                    "y": float(td_pts_results[pname][fidx][jidx][1]) * 100,
                    "z": float(td_pts_results[pname][fidx][jidx][2]) * 100,
                }

        with open(
            os.path.join(BASE_PATH, "depth", os.path.basename(persion_json_path)), "w"
        ) as f:
            json.dump(json_datas, f, indent=4)


if __name__ == "__main__":
    exec()
