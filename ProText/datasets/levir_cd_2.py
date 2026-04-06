from dassl.data.datasets import DATASET_REGISTRY, DatasetBase
from dassl.utils import read_image
import os
import os.path as osp
import numpy as np
from PIL import Image

@DATASET_REGISTRY.register()
class LEVIR_CD(DatasetBase):
    dataset_dir = "LEVIR-CD"

    def __init__(self, cfg, **kwargs):
        root = cfg.DATASET.ROOT
        super().__init__(cfg)
        self.dataset_dir = osp.join(root, "LEVIR-CD")

        # 类别名
        classnames = ["unchanged", "changed"]
        self._lab2cname = {0:"unchanged", 1:"changed"}
        self._cname2lab = {"unchanged":0, "changed":1}

        # 读取数据
        train = self.read_split("train.txt")
        val = self.read_split("val.txt")
        test = self.read_split("test.txt")

        super().__init__(train, val, test, num_shots, classnames)

    def read_split(self, fn):
        items = []
        with open(osp.join(self.dataset_dir, fn),"r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                imgA, imgB, lbl = line.split()
                # 读取label图是否变化
                mask = np.array(Image.open(lbl).convert("L"))
                label = 1 if np.any(mask>0) else 0
                items.append( (imgA, imgB, label) )
        return items

    def __getitem__(self, idx):
        # ✅ 同时返回 A、B、label
        imgA_path, imgB_path, label = self.data[idx]

        imgA = read_image(imgA_path)
        imgB = read_image(imgB_path)

        # 关键：返回 A 和 B
        return imgA, imgB, label, idx