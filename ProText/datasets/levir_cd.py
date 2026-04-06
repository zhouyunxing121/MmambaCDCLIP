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

        print(f"训练集数量: {len(train)}")

        self._train_x = train
        super().__init__(train_x=train, val=val, test=test)

    @property
    def train_x(self):
        return self._train_x

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
            mask = np.array(Image.open(mask_path).convert("L"))
            label = 1 if np.any(mask > 0) else 0
            # 关键：只存 impath，不存 impath2！Dassl不支持！
            items.append(Datum(impath=pathA, label=label))
        return items

    def __getitem__(self, idx):
        item = self._train_x[idx]
        pathA = item.impath
        pathB = pathA.replace("A", "B")  # 自动找B图，不用存impath2

        imgA = read_image(pathA)
        imgB = read_image(pathB)

        # ==================== 自动适配 ====================
        # 训练 ProText：返回 单图 + 标签
        if self.training:
            return imgA, item.label
        # 测试/变化检测：返回 双图 + 标签
        else:
            return imgA, imgB, item.label