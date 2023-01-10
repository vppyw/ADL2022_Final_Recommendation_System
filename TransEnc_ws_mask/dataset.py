import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm

FEATURE_NUM=149
COURSE_NUM=728
SUBGROUP_NUM=91 # 1 ~ 149

class UserDataset(Dataset):
    def __init__(self,
                 file,
                 feature_map_file="feature_map.json",
                 course_map_file="course_map.json",
                 mask_p=0.0,
                 do_seen=False,
                 seen_file=None):
        super().__init__()
        self.mask_p = mask_p
        self.do_seen= do_seen
        with open(file, "r") as f:
            users = json.load(f)
        with open(feature_map_file, "r") as f:
            feature_map = json.load(f)
        with open(course_map_file, "r") as f:
            course_map = json.load(f)
        
        if seen_file != None:
            with open(seen_file, "r") as f:
                seen_datas = json.load(f)
            seen_datas = {
                data["user_id"]:data for data in seen_datas
            }
        else:
            seen_datas = {}
        self.datas = []
        feature_size = len(feature_map.keys()) + 1
        pbar = tqdm(users, ncols=50)
        for user in pbar:
            feature = torch.zeros(feature_size)
            if user["gender"] in feature_map.keys():
                feature[feature_map[user["gender"]]] = 1.0
            for occupation in user["occupation_titles"]:
                if occupation in feature_map.keys():
                    feature[feature_map[occupation]] = 1.0
            for interest in user["interests"]:
                if interest in feature_map.keys():
                    feature[feature_map[interest]] = 1.0
            for recreation in user["recreation_names"]:
                if recreation in feature_map.keys():
                    feature[feature_map[recreation]] = 1.0
            course_size = COURSE_NUM
            course = torch.zeros(course_size)
            for cid in user["course_id"]:
                course[course_map[cid]] = 1.0

            subgroup_size = SUBGROUP_NUM + 1
            subgroup = torch.zeros(subgroup_size)
            subgroup[[user["subgroup"]]] = 1.0

            seen_course = torch.zeros_like(course)
            if user["user_id"] in seen_datas.keys():
                for cid in seen_datas[user["user_id"]]["course_id"]:
                    seen_course[course_map[cid]] = 1.0

            data = (
                user["user_id"],
                feature,
                seen_course,
                course,
                subgroup
            ) 
            self.datas.append(data)

    def __getitem__(self, idx):
        """
        user_id, feature, course, subgroup
        """
        uid, feature, seen_course, course, subgroup = self.datas[idx]
        if self.mask_p > 0:
            feature = torch.where(
                        (feature != 0).logical_and(
                                        torch.rand_like(feature).float()
                                       ),
                        feature,
                        0
                      )
        if self.do_seen:
            idx = seen_course.nonzero().reshape(-1).tolist()
            random.shuffle(idx)
            tar_num = random.randint(1, len(idx))
            seen_idx = idx[tar_num:]
            seen_course = torch.zeros_like(course)
            seen_course[seen_idx] = 1.0
            tar_idx = idx[:tar_num]
            course = torch.zeros_like(course)
            course[tar_idx] = 1.0
        feature = torch.cat([feature, seen_course])
        return uid, feature, course, subgroup

    def __len__(self):
        return len(self.datas)
