import os
import json
import csv
from argparse import ArgumentParser
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers

from tqdm import tqdm

import dataset
import model
from metrics import mAPk

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
        trainset = dataset.UserDataset(file=args.train_file,
                                       mask_p=args.mask_p,
                                       do_seen=True,
                                       seen_file=args.train_file)
        trainset = DataLoader(trainset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    else:
        trainset = None
    if args.valid_file != None:
        validset = dataset.UserDataset(file=args.valid_file,
                                       seen_file=args.train_file)
        validset = DataLoader(validset,
                              batch_size=args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=False)
    else:
        validset = None
    if args.test_file != None:
        testset = dataset.UserDataset(file=args.test_file,
                                      seen_file=args.train_file)
        testset = DataLoader(testset,
                             batch_size=args.batch_size,
                             num_workers=args.num_workers,
                             shuffle=False)
    else:
        testset = None

    with open(args.log_file, "w") as f:
        json.dump([], f, indent=4, ensure_ascii=False)

    if args.load_pretrain_model != None:
        classifier = torch.load(args.load_pretrain_model,
                                map_location=args.device)
    else:
        if args.predict_class == "course":
            classifier = model.TransRec(
                            num_feature=dataset.FEATURE_NUM\
                                        +dataset.COURSE_NUM,
                            output_dim=dataset.COURSE_NUM,
                            hidden_dim=args.hidden_dim,
                            embed_dim=args.embed_dim,
                            nhead=args.nhead,
                            num_layers=args.num_layers,
                            padding_idx=0,
                            dropout=args.dropout,
                         ).to(args.device)
        elif args.predict_class == "subgroup":
            classifier = model.TransRec(
                            num_feature=dataset.FEATURE_NUM\
                                        +dataset.COURSE_NUM,
                            output_dim=dataset.SUBGROUP_NUM+1,
                            hidden_dim=args.hidden_dim,
                            embed_dim=args.embed_dim,
                            nhead=args.nhead,
                            num_layers=args.num_layers,
                            padding_idx=0,
                            dropout=args.dropout,
                         ).to(args.device)

    if args.train_file != None or args.valid_file != None:
        with open(args.model_config_file, "w") as f:
            f.write(classifier.__str__())
        optim = torch.optim.Adam(classifier.parameters(), lr=args.lr)
        if args.do_scheduler:
            scheduler = transformers.get_cosine_schedule_with_warmup(
                            optim,
                            num_warmup_steps=args.warmup_step,
                            num_training_steps=args.epoch,
                        )

    loss_fn = nn.BCEWithLogitsLoss()

    pbar = tqdm(range(args.epoch), ncols=70)
    bst_mAP = float("-inf")
    last_update = 0
    for epoch in pbar:
        train_loss = []
        train_mAP = []
        valid_loss = []
        valid_mAP = []
        if trainset != None:
            classifier.train()
            for uid, feat, course, subgroup in trainset:
                feat = feat.to(args.device)
                predict = classifier(feat)
                if args.predict_class == "course":
                    course = course.to(args.device)
                    loss = loss_fn(predict, course)
                    train_mAP.append(onehot2mAPk(predict,
                                                 course,
                                                 dataset.COURSE_NUM,
                                                 k=50))
                elif args.predict_class == "subgroup":
                    subgroup = subgroup.to(args.device)
                    loss = loss_fn(predict, subgroup)
                    train_mAP.append(onehot2mAPk(predict,
                                                 subgroup,
                                                 0,
                                                 k=50))
                train_loss.append(loss.item())
                optim.zero_grad()
                loss.backward()
                optim.step()
            if args.do_scheduler:
                scheduler.step()
                pbar.set_description(
                    f"{scheduler.get_last_lr()[0]:.4e}"
                )
            train_loss = np.mean(train_loss) \
                            if len(train_loss) > 0 else 0
            train_mAP = np.mean(train_mAP) \
                            if len(train_mAP) > 0 else 0
        if validset != None:
            classifier.eval()
            with torch.no_grad():
                for uid, feat, course, subgroup in validset:
                    feat = feat.to(args.device)
                    predict = classifier(feat)
                    if args.predict_class == "course":
                        course = course.to(args.device)
                        loss = loss_fn(predict, course)
                        valid_mAP.append(onehot2mAPk(predict,
                                                     course,
                                                     dataset.COURSE_NUM,
                                                     k=50))
                    elif args.predict_class == "subgroup":
                        subgroup = subgroup.to(args.device)
                        loss = loss_fn(predict, subgroup)
                        valid_mAP.append(onehot2mAPk(predict,
                                                     subgroup,
                                                     0,
                                                     k=50))
                    valid_loss.append(loss.item())
            valid_loss = np.mean(valid_loss) \
                            if len(valid_loss) > 0 else 0
            valid_mAP = np.mean(valid_mAP) \
                            if len(valid_mAP) > 0 else 0
            if valid_mAP > bst_mAP:
                last_update = epoch
                bst_mAP = valid_mAP
                classifier.eval()
                classifier = classifier.to("cpu")
                torch.save(classifier, args.save_model_file)
                classifier = classifier.to(args.device)
        if validset == None and args.do_save:
            classifier.eval()
            classifier = classifier.to("cpu")
            torch.save(classifier, args.save_model_file)
            classifier = classifier.to(args.device)

        with open(args.log_file, "r") as f:
            log = json.load(f)
        with open(args.log_file, "w") as f:
            log.append({"epoch":epoch,
                        "train_loss":train_loss,
                        "valid_loss":valid_loss,
                        "train_mAP": train_mAP,
                        "valid_mAP": valid_mAP})
            json.dump(log, f, indent=4, ensure_ascii=False)
        
        if args.early_stop != None \
           and epoch - last_update > args.early_stop:
           break

    if testset != None:
        if args.load_model_test != None:
            classifier = torch.load(args.load_model_test,
                                    map_location=args.device)
        elif args.save_model_file != None:
            classifier = torch.load(args.save_model_file,
                                    map_location=args.device)
        classifier.eval()
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
                feat = feat.to(args.device)
                predict = classifier(feat)
                for uid, pred in zip(uids, predict):
                    if args.predict_class == "course":
                        if seen_data != None:
                            vals = predict2cid(pred,
                                               dataset.COURSE_NUM,
                                               idx2cid,
                                               k=50+len(seen_data[uid]))
                            vals = seen_filter(vals, seen_data[uid])
                        else:
                            vals = predict2cid(pred,
                                               dataset.COURSE_NUM,
                                               idx2cid,
                                               k=50)
                    elif args.predict_class == "subgroup":
                        vals = predict2idx(pred,
                                           0,
                                           k=50)
                    csv_writer.writerow([uid, " ".join(map(str, vals))])
        predict_f.close()

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--do_save", action="store_true")
    parser.add_argument("--predict_class", type=str,
                        choices=["course", "subgroup"])
    parser.add_argument("--config_file", type=str, default="config.json")
    parser.add_argument("--train_file", type=str)
    parser.add_argument("--valid_file", type=str)
    parser.add_argument("--test_file", type=str)
    parser.add_argument("--predict_file", type=str)
    parser.add_argument("--seen_filter", type=str, nargs="*")
    parser.add_argument("--load_pretrain_model", type=str)
    parser.add_argument("--load_model_test", type=str)
    parser.add_argument("--save_model_file", type=str, default="model.pt")
    parser.add_argument("--log_file", type=str, default="log.json")
    parser.add_argument("--model_config_file", type=str, default="model.json")

    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--do_scheduler", action="store_true")
    parser.add_argument("--warmup_step", type=int)
    parser.add_argument("--early_stop", type=int)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=1)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--mask_p", type=float, default=0.0)
    args = parser.parse_args()
    return args

def prepare_env(args):
    os.makedirs(args.dir, exist_ok=True)
    args.config_file = os.path.join(args.dir, args.config_file)
    args.log_file = os.path.join(args.dir, args.log_file)
    args.model_config_file = os.path.join(args.dir,
                                          args.model_config_file)
    args.save_model_file = os.path.join(args.dir,
                                        args.save_model_file)
    if args.predict_file != None:
        args.predict_file = os.path.join(args.dir,
                                         args.predict_file)
    with open(args.config_file, "w") as f:
        json.dump(vars(args), f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    args = parse_args()
    prepare_env(args)
    main(args)
