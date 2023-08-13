import os
import tarfile
import webdataset as wds
import json
import torch
from typing import Dict
import numpy as np

if __name__ == '__main__':

    # with open('results/models/upstream/GPT_lrs-4_hds-12_embd-768_train-CSM_lr-0001_bs-64_drp-01_2023-08-05_19-37-12/tarfile_paths_split.json', 'r') as f:
    #     tarfile_paths_split = json.load(f)

    # train_tarfile_paths = tarfile_paths_split['train']
    # validation_tarfile_paths = tarfile_paths_split['validation']
    # test_tarfile_paths = tarfile_paths_split['test'] if 'test' in tarfile_paths_split else None


    # dataset = wds.WebDataset(train_tarfile_paths)
    # dataset = dataset.decode("pil").map(preprocess_sample)

    path = "data/upstream" #文件夹目录
    files= os.listdir(path) #得到文件夹下的所有文件名称
    s = []

    for file in files: #遍历文件夹
        tars = os.listdir(path + '/' + file)
        for tar in tars:
            dataset = wds.WebDataset(path + '/' + file  + '/' + tar)
            for d in dataset:
                out = dict(__key__=d["__key__"])
                t_r = d["t_r.pyd"]

                label = None
                f_s = None
                for key, value in d.items():
                    if key == "bold.pyd":

                        bold = np.array(value).astype(np.float)

                        if self.bold_dummy_mode:
                            bold = self.make_bold_dummy(
                                bold_shape=bold.shape,
                                t_r=t_r,
                                f_s=f_s
                            )

                        seq_on, seq_len = self._sample_seq_on_and_len(bold_len=len(bold))
                        bold = bold[seq_on:seq_on+seq_len]
                        t_rs = np.arange(seq_len) * t_r
                        attention_mask = np.ones(seq_len)
                        bold = self._pad_seq_right_to_n(
                            seq=bold,
                            n=self.seq_max,
                            pad_value=0
                        )
                        t_rs = self._pad_seq_right_to_n(
                            seq=t_rs,
                            n=self.seq_max,
                            pad_value=0
                        )
                        attention_mask = self._pad_seq_right_to_n(
                            seq=attention_mask,
                            n=self.seq_max,
                            pad_value=0
                        )
                        out["inputs"] = torch.from_numpy(bold).to(torch.float)
                        out['t_rs'] = torch.from_numpy(t_rs).to(torch.float)
                        out["attention_mask"] = torch.from_numpy(attention_mask).to(torch.long)
                        out['seq_on'] = seq_on
                        out['seq_len'] = seq_len

                    else:
                        out[key] = value
                        
                bold = d['bold.pyd']

                out = dict(__key__=key["__key__"])
                t_r = key["t_r.pyd"]

                print(key)
                print(data)
            t = tarfile.open()
            if not os.path.isdir(file): #判断是否是文件夹，不是文件夹才打开
                f = open(path+"/"+file); #打开文件
                iter_f = iter(f); #创建迭代器
                str = ""
                for line in iter_f: #遍历文件，一行行遍历，读取文本
                    str = str + line
                s.append(str) #每个文件的文本存到list中

