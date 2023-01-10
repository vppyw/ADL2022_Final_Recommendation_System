import os
import csv
import json
import numpy as np
from argparse import ArgumentParser
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from tqdm import tqdm
import model
import dataset

COURSE_NUM=728
SUBGROUP_NUM=91 # 1 ~ 91

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--do_log", action="store_true")
    parser.add_argument("--predict_class", type=str,
                        choices=["course", "subgroup"])
    parser.add_argument("--config_file", type=str, default="config.json")
    parser.add_argument("--course_file", type=str, required=True)
    parser.add_argument("--course_map", type=str, required=True)
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--predict_file", type=str)
    parser.add_argument("--seen_filter", type=str, nargs="*")
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
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--course_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulate", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--step_lr_step", type=int)
    parser.add_argument("--step_lr_gamma", type=float)
    parser.add_argument("--warmup_step", type=int)
    parser.add_argument("--metrics", type=str, default="mAP")
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
    for select_idx , cid, courses_desc in courseset:
        courses_desc = course_extr(courses_desc, device)
        course_embedding = torch.cat(
                            (course_embedding, courses_desc),
                            dim=0
                           )
    return course_embedding

def predict_step(args,
                 courseset, userset,
                 user_extr, course_extr,
                 do_mAP=True):
    user_extr.eval()
    course_extr.eval()
    with torch.no_grad():
        course_embed = extr_course_embedding(course_extr,
                                             courseset,
                                             args.device)
    mAPk = []
    uids = []
    predicts = torch.Tensor([]).to(args.device)
    for uid, user_desc, course_tar, _ in userset:
        with torch.no_grad():
            user_embed = user_extr(user_desc,
                                   args.device)
            predict = user_embed.mm(
                         course_embed.transpose(0, 1)
                      )
            predicts = torch.cat(
                        [predicts, predict],
                        dim=0
                       )
            uids.extend(uid)
            if do_mAP:
                mAPk.append(onehot2mAPk(predict,
                                        course_tar,
                                        non_idx=model.COURSE_NUM,
                                        k=50))
    return uids, predicts, np.mean(mAPk)

def main(args):
    with open(args.course_map, "r") as f:
        cid2idx = json.load(f)
    idx2cid = {cid2idx[cid]:cid for cid in cid2idx.keys()}
    courseset = dataset.CourseDataset(args.course_file,
                                      max_length=args.max_length)
    courseset_shuffle = DataLoader(courseset,
                                   batch_size=args.course_batch_size,
                                   num_workers=args.num_workers,
                                   shuffle=True)
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
        testset = dataset.UserDataset(user_file=args.test_file,
                                      course_map=args.course_map,
                                      max_length=args.max_length,
                                      do_shuffle=False)
        testset = DataLoader(testset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False)
    else:
        testset = None

    if args.do_log:
        with open(args.log_file, "w") as f:
            json.dump([], f, indent=2, ensure_ascii=False)

    if args.load_pretrain_user_model != None:
        user_extr = torch.load(args.load_pretrain_user_model,
                               map_location="cpu")
    else:
        user_extr = model.BertExtractor(model_path=args.model_path,
                                        max_length=args.max_length)
    user_extr = user_extr.to(args.device)

    if args.load_pretrain_course_model != None:
        course_extr = torch.load(args.load_pretrain_course_model,
                                 map_location="cpu")
    else:
        course_extr = model.BertExtractor(model_path=args.model_path,
                                          max_length=args.max_length)
    course_extr = course_extr.to(args.device)

    optim = torch.optim.Adam(
                [
                    {"params": user_extr.parameters(),
                     "lr": args.lr},
                    {"params": course_extr.parameters(),
                     "lr": args.lr},
                ],
                lr=args.lr)
    if args.warmup_step != None:
        lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                            optim,
                            num_warmup_steps=args.warmup_step,
                            num_training_steps=args.epoch\
                                               * (len(trainset)/args.batch_size+1)\
                                               * (len(courseset)/args.course_batch_size+1)\
                                               / args.gradient_accumulate
                       )
    elif args.step_lr_step != None\
        and args.step_lr_gamma != None:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
                            optim,
                            step_size=args.step_lr_step,
                            gamma=args.step_lr_gamma,
                       )

    loss_fn = nn.BCEWithLogitsLoss()

    tot_step = 0
    bst_AP = float("-inf")
    for epoch in range(args.epoch):
        train_loss = []
        if trainset != None:
            pbar = tqdm(trainset, ncols=50)
            loss = 0
            for u_step, (uid, user_desc, course_tar, _)\
                in enumerate(pbar):
                user_extr.train()
                course_extr.train()
                user_embed = user_extr(user_desc,
                                       args.device)
                for c_step, (select_idx, cid, courses_desc)\
                    in enumerate(courseset_shuffle):
                    course_embed = course_extr(
                                    courses_desc,
                                    args.device
                                   )
                    predict = user_embed.mm(
                                course_embed.transpose(0, 1)
                              )
                    select_idx = select_idx\
                                    .squeeze()\
                                    .to(torch.int64)\
                                    .to(args.device)
                    course_tar = course_tar.to(args.device)
                    select_course = course_tar[:,select_idx]
                    loss += loss_fn(predict, select_course)
                    if (c_step + 1) % args.gradient_accumulate == 0 or \
                        c_step == len(courseset) - 1:
                        loss.backward()
                        optim.step()
                        lr_scheduler.step()
                        optim.zero_grad()
                        train_loss.append(loss.item())
                        loss = 0
                        user_embed = user_extr(user_desc,
                                               args.device)
                        tot_step += 1
                if (u_step + 1) % args.steps == 0 or\
                    u_step == len(trainset) - 1:
                    if args.do_log:    
                        with open(args.log_file, "r") as f:
                            log = json.load(f)
                    if validset != None:
                        uids, predicts, valid_mAP = predict_step(
                            args, courseset, validset,
                            user_extr, course_extr
                        )
                        if valid_mAP > bst_AP:
                            bst_AP = valid_mAP
                            if args.do_save:
                                torch.save(user_extr,
                                           args.save_user_model_file)
                                torch.save(course_extr,
                                           args.save_course_model_file)
                        log.append({
                            "tot_step": tot_step,
                            "train_loss": np.mean(train_loss),
                            "valid_mAP": valid_mAP
                        })
                    else:
                        if args.do_save:
                            torch.save(user_extr,
                                       args.save_user_model_file)
                            torch.save(course_extr,
                                       args.save_course_model_file)
                            log.append({
                                "tot_step": tot_step,
                                "train_loss": np.mean(train_loss),
                            })
                    train_loss = []
                    if args.do_log:
                        with open(args.log_file, "w") as f:
                            json.dump(log, f,
                                      indent=2,
                                      ensure_ascii=False)
    if testset != None:
        if args.load_user_model_test != None:
            user_extr = torch.load(args.load_user_model_test,
                                    map_location=args.device)
        else:
            user_extr = torch.load(args.save_user_model_file,
                                   map_location=args.device)
        user_extr.eval()
        if args.load_course_model_test != None:
            course_extr = torch.load(args.load_course_model_test,
                                     map_location=args.device)
        else:
            course_extr = torch.load(args.save_course_model_file,
                                     map_location=args.device)
        course_extr.eval()

        uids, predicts, valid_mAP = predict_step(
            args, courseset, testset,
            user_extr, course_extr,
            do_mAP=False,
        )

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

        for uid, predict in zip(uids, predicts):
            if seen_data != None:
                vals = predict2cid(predict,
                                   non_idx=model.COURSE_NUM,
                                   idx2cid=idx2cid,
                                   k=50+len(seen_data[uid]))
                vals = seen_filter(vals, seen_data[uid])
            else:
                vals = predict2cid(predict,
                                   non_idx=model.COURSE_NUM,
                                   idx2cid=idx2cid,
                                   k=50)
            csv_writer.writerow([uid, " ".join(map(str, vals))])
        predict_f.close()

if __name__ == "__main__":
    args = parse_args()
    prepare_env(args)
    main(args)
