from dassl.utils import read_image
import os
import os.path as osp
import numpy as np
from PIL import Image

import pickle
from collections import OrderedDict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import listdir_nohidden, mkdir_if_missing

from .oxford_pets import OxfordPets


@DATASET_REGISTRY.register()
class LEVIR_CD(DatasetBase):
    """
    LEVIR-CD 变化检测数据集加载类
    目录结构：
    LEVIR-CD/
    ├── train/
    │   ├── A/ (时相1图像)
    │   ├── B/ (时相2图像)
    │   └── label/ (二值变化标签)
    ├── val/
    ├── test/
    ├── train.txt
    ├── val.txt
    └── test.txt
    txt格式：A路径  B路径  label路径
    """
    dataset_dir = "LEVIR-CD"

    def __init__(self, cfg, **kwargs):
        root = cfg.DATASET.ROOT
        super().__init__(cfg)
        self.dataset_dir = osp.join(root, "LEVIR-CD")
        
        # 1. 读取类别名（变化检测二分类：no change / change）
        classnames = []
        classname_path = osp.join(root, "classnames", "levir_cd.txt")
        with open(classname_path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    classnames.append(line)
        assert len(classnames) == 2, f"类别数错误，应为2，实际{len(classnames)}"
        self._lab2cname = {i: name for i, name in enumerate(classnames)}
        self._cname2lab = {name: i for i, name in enumerate(classnames)}

        # 2. 读取train/val/test数据
        train = self._read_split("train.txt")
        val = self._read_split("val.txt")
        test = self._read_split("test.txt")

        # 3. 调用父类初始化（Dassl标准格式）
        super().__init__(train, val, test, num_shots, classnames)

    def _read_split(self, split_file):
        """
        读取train.txt/val.txt/test.txt
        每行格式：A路径  B路径  label路径
        返回格式：[(impath_A, impath_B, label), ...]
        """
        items = []
        split_path = osp.join(self.dataset_dir, split_file)
        
        with open(split_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                # 分割路径：A路径 B路径 label路径
                parts = line.split()
                assert len(parts) == 3, f"行格式错误：{line}"
                impath_A, impath_B, impath_label = parts

                # 计算标签：label图中是否有非0像素（1=change，0=no change）
                label = self._get_label_from_mask(impath_label)

                # 存储数据项（Dassl标准格式：(impath, label, domain)，这里domain=0）
                # 对于变化检测双时相，我们将A、B路径合并存储，在__getitem__中读取
                items.append(( (impath_A, impath_B), label, 0 ))

        return items

    def _get_label_from_mask(self, mask_path):
        """
        从label图中计算标签：
        - 若存在像素>0 → 判定为change（label=1）
        - 全0 → 判定为no change（label=0）
        """
        # 读取label图（PIL → numpy）
        mask = np.array(Image.open(mask_path).convert("L"))
        # 二值化（LEVIR-CD label为0/255）
        if np.any(mask > 0):
            return 1  # change
        else:
            return 0  # no change

    def __getitem__(self, idx):
        """
        Dassl数据集标准__getitem__方法
        返回：(image_A, image_B, label, index)
        """
        (impath_A, impath_B), label, _ = self.data[idx]

        # 读取双时相图像
        image_A = read_image(impath_A)
        image_B = read_image(impath_B)

        # 对于ProText文本监督，我们将双时相图像拼接为模型输入
        # 若你使用的是单输入模型，可改为：image = torch.cat([image_A, image_B], dim=0)
        return image_A, image_B, label, idx