from dassl.data.datasets import DatasetBase
import os
import os.path as osp

class ChangeDetection(DatasetBase):
    dataset_dir = "change_detection"

    def __init__(self, root, num_shots):
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.image_dir = osp.join(self.dataset_dir, "images")
        self.split_dir = osp.join(self.dataset_dir, "splits")

        # 读取 classnames
        classnames = []
        with open(osp.join(root, "classnames", "change_detection.txt"), "r") as f:
            for line in f:
                classnames.append(line.strip())
        self._lab2cname = {i: name for i, name in enumerate(classnames)}
        self._cname2lab = {name: i for i, name in enumerate(classnames)}

        # 读取数据（按你的数据集结构实现）
        train = self._read_data(osp.join(self.split_dir, "train.txt"))
        val = self._read_data(osp.join(self.split_dir, "val.txt"))
        test = self._read_data(osp.join(self.split_dir, "test.txt"))

        super().__init__(train, val, test, num_shots, classnames)

    def _read_data(self, path):
        data = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                impath, label = line.split()
                label = int(label)
                impath = osp.join(self.image_dir, impath)
                data.append((impath, label))
        return data