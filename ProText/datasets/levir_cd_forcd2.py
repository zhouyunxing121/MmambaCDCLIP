from dassl.data.datasets import DATASET_REGISTRY, DatasetBase, Datum
from dassl.utils import read_image
import os
import os.path as osp
import numpy as np
from PIL import Image

@DATASET_REGISTRY.register()
class LEVIR_CD(DatasetBase):
    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = osp.join(root, "LEVIR-CD")

        train = self._read_data("train.txt")
        val = self._read_data("val.txt")
        test = self._read_data("test.txt")

        print(f"✅ 训练集数量: {len(train)}")
        super().__init__(train_x=train, val=val, test=test)

    @property
    def num_classes(self):
        return 2

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
            # 读取mask判断变化标签
            mask = np.array(Image.open(mask_path).convert("L"))
            label = 1 if np.any(mask > 0) else 0
            # 存储A、B路径 + 标签
            items.append(Datum(impath=pathA, impath2=pathB, label=label))
        return items

    def __getitem__(self, idx):
        item = self.train_x[idx]
        # 独立读取A、B图像
        imgA = read_image(item.impath)
        imgB = read_image(item.impath2)
        # 返回：A图像、B图像、变化标签
        return imgA, imgB, item.label