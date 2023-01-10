import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import json
import random

COURSE_NUM=728
SUBGROUP_NUM=149 # 1 ~ 149

class UserDataset(Dataset):
    def __init__(self,
                 user_file,
                 course_map,
                 max_length=512,
                 do_shuffle=True):
        with open(user_file, "r") as f:
            self.users_data = json.load(f)

        with open(course_map, "r") as f:
            self.course_map = json.load(f)
        self.max_length = max_length
        self.do_shuffle = do_shuffle
        self.gender_dict = {'male': '男生',
                            'female': '女生',
                            'other': '非二元性別',
                            '':'不願透漏性別'}
        
    def __getitem__(self, idx):
        """
        return:
            user description
            course(one hot)
            subgroup(one hot)
        """
        user_data = self.users_data[idx]
        user = (f"我是{user_data['gender']}，" if user_data["gender"] in self.gender_dict else "") + \
                "我的職業是" + \
                "、".join(user_data["occupation_titles"]) + \
                "，我的興趣是" + \
                "、".join(user_data["recreation_names"]) + \
                "，我想要學習"
        interests = user_data["interests"]
        if self.do_shuffle:
            random.shuffle(interests)
        for interest in interests:
            if len(user) + len(interest) >= self.max_length:
                break
            user += interest + "、"
        user = user[:-1]
        course_idx = list(self.course_map[cid] \
                          for cid in user_data["course_id"])
        course = torch.zeros(COURSE_NUM)
        course[[course_idx]] = 1
                 
        subgroup_idx = user_data["subgroup"]
        subgroup = torch.zeros(SUBGROUP_NUM)
        subgroup[[subgroup_idx]] = 1
        return user_data["user_id"], user, course, subgroup

    def __len__(self):
        return len(self.users_data)

class CourseDataset(Dataset):
    def __init__(self, course_file, max_length=512):
        with open(course_file, "r") as f:
            self.courses_data = json.load(f)
        self.max_length = max_length
    
    def __getitem__(self, idx):
        course_data = self.courses_data[idx]
        course = course_data["course_name"] + "是一堂跟" + \
                 "、".join(course_data["groups"]) + "以及" + \
                 "、".join(course_data["sub_groups"]) + \
                 "相關的課程。" + \
                 "課程的主題是" + "、".join(course_data["topics"]) + \
                 "，在這堂課你會學到" + course_data["will_learn"] + "。" + \
                 "這堂課需要" + course_data["course_price"] + "元，適合" + \
                 course_data["target_group"] + "的人，並且建議" + \
                 course_data["recommended_background"] + "。" + \
                 "上課可能會需要用到" + course_data["required_tools"] + "。"
        course = course[:self.max_length]
        return idx, course_data["course_id"], course
        
    def __len__(self):
        return len(self.courses_data)
