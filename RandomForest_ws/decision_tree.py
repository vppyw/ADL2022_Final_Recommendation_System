import os
import random
import pickle
import joblib
import json
import csv
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate 
from tqdm import tqdm

from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

from metrics import mAPk

FEATURE_NUM=149
COURSE_NUM=728
SUBGROUP_NUM=91 # 1 ~ 149

class UserDataset(Dataset):
    def __init__(self,
                 file,
                 feature_map_file="feature_map.json",
                 course_map_file="course_map.json",
                 mask_num=0,
                 mask_p=0.0,
                 seen_file=None):
        super().__init__()
        with open(file, "r") as f:
            users = json.load(f)
        with open(feature_map_file, "r") as f:
            feature_map = json.load(f)
        with open(course_map_file, "r") as f:
            course_map = json.load(f)

        if seen_file != None:
            with open(seen_file, "r") as f:
                seen_datas = json.load(f)
                seen_datas = {data["user_id"]:data for data in seen_datas}
        else:
            seen_datas = {}

        self.datas = []
        feature_size = len(feature_map.keys()) + 4
        pbar = tqdm(users, ncols=50)
        for user in pbar:
            feature = torch.zeros(feature_size)
            if user["gender"] in feature_map.keys():
                feature[feature_map[user["gender"]]] = 1.0
            else:
                feature[feature_size - 4] = 1.0
            for occupation in user["occupation_titles"]:
                if occupation in feature_map.keys():
                    feature[feature_map[occupation]] = 1.0
                else:
                    feature[feature_size - 3] = 1.0
            for interest in user["interests"]:
                if interest in feature_map.keys():
                    feature[feature_map[interest]] = 1.0
                else:
                    feature[feature_size - 2] = 1.0
            for recreation in user["recreation_names"]:
                if recreation in feature_map.keys():
                    feature[feature_map[recreation]] = 1.0
                else:
                    feature[feature_size - 1] = 1.0

            course_size = COURSE_NUM
            course = torch.zeros(course_size)
            for cid in user["course_id"]:
                course[course_map[cid]] = 1.0

            subgroup_size = SUBGROUP_NUM + 1
            subgroup = torch.zeros(subgroup_size)
            flag = True
            for grp in user["subgroup"]:
                flag = False
                subgroup[grp] = 1.0
            if flag:
                subgroup[0] = 1.0

            if mask_num == 0:
                seen_course = torch.zeros_like(course)
                if user["user_id"] in seen_datas.keys():
                    for cid in seen_datas[user["user_id"]]["course_id"]:
                        seen_course[course_map[cid]] = 1.0
                data = (
                    user["user_id"],
                    torch.cat([feature, seen_course]),
                    course,
                    subgroup,
                )
                self.datas.append(data)
            else:
                idx = course.nonzero().reshape(-1).tolist()
                for _ in range(mask_num):
                    random.shuffle(idx)
                    tar_num = min(len(idx), round(len(idx) * mask_p) + 1)
                    seen_course = torch.zeros(course_size)
                    seen_course[idx[tar_num:]] = 1.0
                    course = torch.zeros(course_size)
                    course[idx[:tar_num]] = 1.0
                    data = (
                        user["user_id"],
                        torch.cat([feature, seen_course]),
                        course,
                        subgroup,
                    )
                    self.datas.append(data)
    def __getitem__(self, idx):
        """
        user_id, feature, course, subgroup
        """
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)

def collate_fn(batch):
    uid = [data[0] for data in batch]
    _batch = [data[1:] for data in batch]
    return uid, *default_collate(_batch)

def predict2idx(predict, non_idx, k=50):
    if predict.dim() != 1:
        print("predict should be dim=1!!!")
        raise RuntimeError
    vals, idices = predict.topk(min(k, predict.size(0)))
    idices = idices.tolist()
    if non_idx in idices:
        idices.remove(non_idx)
    return idices

def predict2cid(predict, non_idx, idx2cid, k=50):
    idices = predict2idx(predict, non_idx, k)
    cids = [idx2cid[idx] for idx in idices]
    return cids

def onehot2mAPk(predicts, targets, threshold_idx, k=50):
    predict_list = []
    targets_list = []
    for predict, target in zip(predicts, targets):
        predict_list.append(predict2idx(predict, threshold_idx, k))
        targets_list.append((target >= 1).nonzero().reshape(-1).tolist())
    return mAPk(predict_list, targets_list)

def seen_filter(predicts, seen_val):
    ret = []
    for predict in predicts:
        if predict not in seen_val:
            ret.append(predict)
    return ret

def main(args):
    with open("course_map.json", "r") as f:
        cid2idx = json.load(f)
    idx2cid = {cid2idx[cid]:cid for cid in cid2idx.keys()}
    if args.train_file != None:
        trainset = UserDataset(file=args.train_file,
                               mask_num=args.mask_num,
                               mask_p=args.mask_p,
                               seen_file=args.train_file)
        trainset = DataLoader(trainset,
                              batch_size=len(trainset),
                              num_workers=args.num_workers,
                              shuffle=True)
    else:
        trainset = None
    if args.valid_file != None:
        validset = UserDataset(file=args.valid_file,
                               seen_file=args.train_file)
        validset = DataLoader(validset,
                              batch_size=len(validset),
                              num_workers=args.num_workers,
                              shuffle=False)
    else:
        validset = None
    if args.test_file != None:
        testset = UserDataset(file=args.test_file,
                              seen_file=args.train_file)
        testset = DataLoader(testset,
                             batch_size=len(testset),
                             num_workers=args.num_workers,
                             shuffle=False)
    else:
        testset = None

    with open(args.log_file, "w") as f:
        json.dump([], f, indent=4, ensure_ascii=False)

    # classifier = RandomForestRegressor(n_jobs=args.num_workers)
    if args.gpu != None:
        classifier = xgb.XGBRFRegressor(n_jobs=args.num_workers,
                                        gpu_id=args.gpu,
                                        tree_method="gpu_hist")
    else:
        classifier = xgb.XGBRFRegressor(n_jobs=args.num_workers)

    print("=========START TRAIN===========")
    train_mAP = []
    train_score = []
    if trainset != None:
        for uid, feat, course, subgroup in trainset:
            if args.predict_class == "course":
                feat = feat.numpy()
                course = course.numpy()
                classifier.fit(feat, course)
                train_score.append(classifier.score(feat, course))
                predict = torch.from_numpy(classifier.predict(feat))
                course = torch.from_numpy(course)
                train_mAP.append(onehot2mAPk(predict, course, 728, k=50))
            elif args.predict_class == "subgroup":
                feat = feat.numpy()
                subgroup = subgroup.numpy()
                classifier.fit(feat, subgroup)
                train_score.append(classifier.score(feat, subgroup))
                predict = torch.from_numpy(classifier.predict(feat))
                subgroup = torch.from_numpy(subgroup)
                train_mAP.append(onehot2mAPk(predict, subgroup, 0, k=50))
    train_mAP = np.mean(train_mAP) if train_mAP != [] else 0
    train_score = np.mean(train_score) if train_score != [] else 0
    print("=========END TRAIN===========")
    print("=========START VALID===========")
    if args.load_test:
        classifier = joblib.load(args.load_test)
    valid_mAP = []
    valid_score = []
    if validset != None:
        for uid, feat, course, subgroup in validset:
            if args.predict_class == "course":
                feat = feat.numpy()
                course = course.numpy()
                predict = classifier.predict(feat)
                valid_score.append(classifier.score(feat, course))
                predict = torch.from_numpy(predict)
                course = torch.from_numpy(course)
                valid_mAP.append(onehot2mAPk(predict,
                                             course,
                                             728,
                                             k=50))
            elif args.predict_class == "subgroup":
                feat = feat.numpy()
                subgroup = subgroup.numpy()
                predict = classifier.predict(feat)
                valid_score.append(classifier.score(feat, subgroup))
                predict = torch.from_numpy(predict)
                subgroup = torch.from_numpy(subgroup)
                valid_mAP.append(onehot2mAPk(predict,
                                             subgroup,
                                             0,
                                             k=50))
                

    valid_mAP = np.mean(valid_mAP) if len(valid_mAP) != 0 else 0
    valid_score = np.mean(valid_score) if len(valid_score) != 0 else 0
    print("=========END VALID===========")
    if args.do_save:
        with open(args.model_file, "wb") as f:
            joblib.dump(classifier, f)
    if args.load_test:
        with open(args.load_test, "rb") as f:
            classifier = pickle.load(f)

    with open(args.log_file, "w") as f:
        log = {"train_mAP": train_mAP,
               "train_score": train_score,
               "valid_mAP": valid_mAP,
               "valid_score": valid_score}
        json.dump(log, f, indent=4, ensure_ascii=False)
    print("=========START TEST===========")
    if testset != None:
        predict_f = open(args.predict_file, "w")
        csv_writer = csv.writer(predict_f)
        if args.predict_class == "course":
            csv_writer.writerow(["user_id", "course_id"])
        elif args.predict_class == "subgroup":
            csv_writer.writerow(["user_id", "subgroup"])
        if args.seen_filter != None:
            seen_data = {}
            for fname in args.seen_filter:
                with open(fname, "r") as f:
                    f_datas = json.load(f)
                for f_data in f_datas:
                    if f_data["user_id"] in seen_data.keys():
                        seen_data[f_data["user_id"]].extend(f_data["course_id"])
                    else:
                        seen_data[f_data["user_id"]] = f_data["course_id"]
        else:
            seen_data = None

        with torch.no_grad():
            for uids, feat, course, subgroup in testset:
                feat = feat.numpy()
                predict = classifier.predict(feat)
                predict = torch.from_numpy(predict)
                for uid, pred in zip(uids, predict):
                    if args.predict_class == "course":
                        if seen_data != None:
                            vals = predict2cid(pred,
                                               728,
                                               idx2cid,
                                               k=50+len(seen_data[uid]))
                            vals = seen_filter(vals, seen_data[uid])
                        else:
                            vals = predict2cid(pred,
                                               728,
                                               idx2cid,
                                               k=50)
                    elif args.predict_class == "subgroup":
                        vals = predict2idx(pred,
                                           0,
                                           k=50)
                    csv_writer.writerow([uid, " ".join(map(str, vals))])
        predict_f.close()
    print("=========END TEST===========")

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--num_workers", type=int, default=8)

    parser.add_argument("--n_estimators", type=int, default=100)

    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--model_file", type=str, default="model.pt")
    parser.add_argument("--load_test", type=str)
    parser.add_argument("--predict_class", type=str,
                        choices=["course", "subgroup"])
    parser.add_argument("--config_file", type=str, default="config.json")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--predict_file", type=str)
    parser.add_argument("--seen_filter", type=str, nargs="*")
    parser.add_argument("--log_file", type=str, default="log.json")
    parser.add_argument("--mask_num", type=int, default=1)
    parser.add_argument("--mask_p", type=float, defualt=1.0)

    args = parser.parse_args()
    return args

def prepare_env(args):
    os.makedirs(args.dir, exist_ok=True)
    args.config_file = os.path.join(args.dir, args.config_file)
    args.log_file = os.path.join(args.dir, args.log_file)
    if args.predict_file != None:
        args.predict_file = os.path.join(args.dir,
                                         args.predict_file)
    if args.model_file != None:
        args.model_file = os.path.join(args.dir,
                                       args.model_file)
    with open(args.config_file, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    prepare_env(args)
    main(args)
