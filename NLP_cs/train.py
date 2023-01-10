import os
import csv
import json
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import model
import dataset

COURSE_NUM=728
SUBGROUP_NUM=91 # 1 ~ 149

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--predict_class", type=str,
                        choices=["course", "subgroup"])
    parser.add_argument("--config_file", type=str, default="config.json")
    parser.add_argument("--course_file", type=str, required=True)
    parser.add_argument("--course_map", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--predict_file", type=str)
    parser.add_argument("--seen_filter", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--load_pretrain_user_model", type=str)
    parser.add_argument("--load_pretrain_course_model", type=str)
    parser.add_argument("--load_user_model_test", type=str)
    parser.add_argument("--load_course_model_test", type=str)
    parser.add_argument("--save_user_model_file", type=str, default="model_user.pt")
    parser.add_argument("--save_course_model_file", type=str, default="model_course.pt")
    parser.add_argument("--log_file", type=str, default="log.json")
    parser.add_argument("--model_config_file", type=str, default="model.json")

    parser.add_argument("--epoch", type=int, default=3)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--course_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    return args

def prepare_env(args):
    os.makedirs(args.dir, exist_ok=True)
    args.config_file = os.path.join(args.dir, args.config_file)
    args.log_file = os.path.join(args.dir, args.log_file)
    args.model_config_file = os.path.join(args.dir,
                                          args.model_config_file)
    args.save_user_model_file = os.path.join(args.dir,
                                             args.save_user_model_file)
    args.save_course_model_file = os.path.join(args.dir,
                                               args.save_course_model_file)
    if args.predict_file != None:
        args.predict_file = os.path.join(args.dir,
                                         args.predict_file)
    with open(args.config_file, "w") as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)

def APk(predict, target, k=50):
    predict = predict[:k]
    score = 0.0
    num_hits = 0.0
    for idx, p in enumerate(predict):
        if p in target and p not in predict[:idx]:
            num_hits += 1.0
            score += num_hits / (idx + 1.0)
    if not target:
        return 0.0
    return score / min(len(target), k)

def mAPk(predict, target, k=50):
    """
    predict:[instance num, predicted class number for each instance]
        a list of lists of prediction classes
    target:[instance num, target class number for each instance]
        a list of lists of target classes
    """
    return np.mean([APk(p, t, k) for p, t in zip(predict, target)])

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

def onehot2mAPk(predicts, targets, non_idx, k=50):
    predict_list = []
    targets_list = []
    for predict, target in zip(predicts, targets):
        predict_list.append(predict2idx(predict, non_idx, k))
        targets_list.append((target >= 1).nonzero().reshape(-1).tolist())
    return mAPk(predict_list, targets_list)

def seen_filter(predicts, seen_val):
    ret = []
    for predict in predicts:
        if predict not in seen_val:
            ret.append(predict)
    return ret
def extr_course_embedding(course_extr, courseset, device):
    course_embedding = torch.Tensor([]).to(device)
    for cid, courses_desc in courseset:
        courses_desc = course_extr(courses_desc, device)
        course_embedding = torch.cat(
                            (course_embedding, courses_desc),
                            dim=0
                           )
    return course_embedding

def main(args):
    with open(args.course_map, "r") as f:
        cid2idx = json.load(f)
    idx2cid = {cid2idx[cid]:cid for cid in cid2idx.keys()}
    courseset = dataset.CourseDataset(args.course_file,
                                      max_length=args.max_length)
    courseset = DataLoader(courseset,
                           batch_size=args.course_batch_size,
                           num_workers=args.num_workers,
                           shuffle=False)
    if args.train_file != None:
        trainset = dataset.UserDataset(user_file=args.train_file,
                                       course_map=args.course_map,
                                       max_length=args.max_length,
                                       do_shuffle=True)
        trainset = DataLoader(trainset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    else:
        trainset = None
    if args.valid_file != None:
        validset = dataset.UserDataset(user_file=args.valid_file,
                                       course_map=args.course_map,
                                       max_length=args.max_length,
                                       do_shuffle=False)
        validset = DataLoader(validset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False)
    else:
        validset = None
    if args.test_file != None:
        testset = dataset.UesrDataset(user_file=args.test_file,
                                      course_map=args.course_map,
                                      max_length=args.max_length,
                                      do_shuffle=False)
        testset = DataLoader(testset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False)
    else:
        testset = None

    with open(args.log_file, "w") as f:
        json.dump([], f, indent=2, ensure_ascii=False)

    if args.load_pretrain_user_model != None:
        user_extr = torch.load(args.load_pretrain_user_model,
                               map_location="cpu")
    else:
        user_extr = model.XLMRobertaExtractor(model_path=args.model_path,
                                              max_length=args.max_length)
    user_extr = user_extr.to(args.device)

    if args.load_pretrain_course_model != None:
        course_extr = torch.load(args.load_pretrain_course_model,
                                 map_location="cpu")
    else:
        course_extr = model.XLMRobertaExtractor(model_path=args.model_path,
                                                max_length=args.max_length)
    course_extr = course_extr.to(args.device)

    optim = torch.optim.Adam(list(course_extr.pooling.parameters()) + \
                             list(course_extr.MLP.parameters()) + \
                             list(user_extr.pooling.parameters()) + \
                             list(user_extr.MLP.parameters()), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()
              
    steps = 0
    for epoch in range(args.epoch):
        train_loss = []
        if trainset != None:
            pbar = tqdm(trainset, ncols=50)
            for uid, user_desc, course, _ in pbar:
                user_extr.train()
                course_extr.train()
                course_embedding = extr_course_embedding(
                                       course_extr=course_extr,
                                       courseset=courseset,
                                       device=args.device
                                   )
                user_embedding = user_extr(user_desc, args.device)
                course = course.to(args.device)
                cos_sim = user_embedding.mm(
                            course_embedding.transpose(0, 1)
                          )
                loss = loss_fn(cos_sim, course)
                optim.zero_grad()
                loss.backward()
                optim.step()
                steps += 1
                train_loss.append(loss.item())
                if (steps + 1) % args.steps == 0:
                    valid_mAP = []
                    if validset != None:
                        user_extr.eval()
                        course_extr.eval()
                        with torch.no_grad():
                            course_embedding = extr_course_embedding(
                                                   course_extr=course_extr,
                                                   courseset=courseset,
                                                   device=args.device
                                               )
                            for uid, user_desc, course, _ in validset:
                                user_embedding = user_extr(user_desc, args.device)
                                course = course.to(args.device)
                                cos_sim = user_embedding.mm(
                                            course_embedding.transpose(0, 1)
                                          )

                                valid_mAP.append(
                                    onehot2mAPk(
                                        cos_sim,
                                        course,
                                        non_idx=COURSE_NUM,
                                        k=50,
                                    )
                                )
                    with open(args.log_file, "r") as f:
                        log = json.load(f)
                    with open(args.log_file, "w") as f:
                        train_loss = np.mean(train_loss)
                        valid_mAP = np.mean(valid_mAP)
                        log.append({"step":steps,
                                    "train_loss":train_loss,
                                    "valid_mAP": valid_mAP})
                        json.dump(log, f, indent=4, ensure_ascii=False)
                    train_loss = []
                    pbar.write(f"Save log at {steps}")
    if testset != None:
        user_extr.eval()
        course_extr.eval()
        with torch.no_grad():
            pass

if __name__ == "__main__":
    args = parse_args()
    prepare_env(args)
    main(args)
