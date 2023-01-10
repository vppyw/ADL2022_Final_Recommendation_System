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
                 train=True):
        super().__init__()
        with open(file, "r") as f:
            users = json.load(f)
        with open(feature_map_file, "r") as f:
            feature_map = json.load(f)
        with open(course_map_file, "r") as f:
            course_map = json.load(f)

        self.train = train
        self.datas = []
        feature_size = len(feature_map.keys()) + 1
        pbar = tqdm(users, ncols=50)
        for user in pbar:
            feature = torch.zeros(feature_size)
            if user["gender"] in feature_map.keys():
                feature[feature_map[user["gender"]]] = feature_map[user["gender"]]
            for occupation in user["occupation_titles"]:
                if occupation in feature_map.keys():
                    feature[feature_map[occupation]] = feature_map[occupation]
            for interest in user["interests"]:
                if interest in feature_map.keys():
                    feature[feature_map[interest]] = feature_map[interest]
            for recreation in user["recreation_names"]:
                if recreation in feature_map.keys():
                    feature[feature_map[recreation]] = feature_map[recreation]
            course_size = COURSE_NUM
            course = torch.zeros(course_size)
            for cid in user["course_id"]:
                course[course_map[cid]] = 1.0

            subgroup_size = SUBGROUP_NUM + 1
            subgroup = torch.zeros(subgroup_size)
            subgroup[[user["subgroup"]]] = 1.0

            data = (user["user_id"], feature, course, subgroup) 
            self.datas.append(data)

    def __getitem__(self, idx):
        """
        user_id, feature, course, subgroup
        """
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)
