from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import read_image
import os
import os.path as osp
import numpy as np
from PIL import Image
import torch
from torchvision import transforms

@DATASET_REGISTRY.register()
class LEVIR_CD(DatasetBase):
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, "LEVIR-CD")

        train = self._read_data("train.txt")
        val = self._read_data("val.txt")
        test = self._read_data("test.txt")

        print(f"✅ 训练集数量: {len(train)}")

        self._train_x = train
        super().__init__(train_x=train, val=val, test=test)

    @property
    def train_x(self):
        return self._train_x

    @property
    def num_classes(self):
        return 2

    # ✅ 【修复关键1】和 JSON、txt 完全统一！
    @property
    def classnames(self):
        return ["unchanged", "changed"]

    def _read_data(self, filename):
        items = []
        file_path = osp.join(self.dataset_dir, filename)
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            pathA, pathB, mask_path = line.split()
            mask = np.array(Image.open(mask_path).convert("L"))
            label = 1 if np.any(mask > 0) else 0
            items.append(Datum(impath=pathA, label=label))
        return items

    def __getitem__(self, idx):
        item = self._train_x[idx]
        pathA = item.impath
        pathB = pathA.replace("A", "B")
        
        imgA = read_image(pathA)
        imgB = read_image(pathB)

        # ==============================
        # ✅ 【修复关键2】融合双时相图像 → 3通道，适配CLIP
        # ==============================
        to_tensor = transforms.ToTensor()
        imgA = to_tensor(imgA)
        imgB = to_tensor(imgB)
        
        # 平均融合，保持 3 通道
        img = (imgA + imgB) / 2.0

        # ✅ 【修复关键3】ProText 只接受 2 个返回值！
        return img, item.label