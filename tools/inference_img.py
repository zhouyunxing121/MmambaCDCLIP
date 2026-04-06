# Copyright (c) OpenMMLab. All rights reserved.
from argparse import ArgumentParser
import torch
from mmengine.model import revert_sync_batchnorm
from mmengine.dataset import Compose
from collections import defaultdict
import numpy as np
import cv2
import os
from mmseg.apis import init_model
import tqdm
from mmseg.models.data_preprocessor import SegDataPreProcessor


def _preprare_data(imgs, model):

    cfg = model.cfg
    for t in cfg.test_pipeline:
        if t.get('type') == 'LoadAnnotations':
            cfg.test_pipeline.remove(t)

    imgs = [imgs]
    is_batch = False

    # TODO: Consider using the singleton pattern to avoid building
    # a pipeline for each inference
    pipeline = Compose(cfg.test_pipeline)

    data = defaultdict(list)
    for img in imgs:
        data_ = dict(img_path=img[0], img_path2=img[1])
        data_ = pipeline(data_)
        data['inputs'].append(data_['inputs'])
        data['data_samples'].append(data_['data_samples'])

    return data, is_batch


def main():
    parser = ArgumentParser()
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    if args.device == 'cpu':
        model = revert_sync_batchnorm(model)

    # test a pair if images
    imgA_path = '/home/wangshiying/gjx/rschange/data/SYSU-CD/test/time1/03414.png'
    imgB_path = '/home/wangshiying/gjx/rschange/data/SYSU-CD/test/time2/03414.png'
    jsonA = {
        "image_path": "/home/wangshiying/gjx/rschange/data/SYSU-CD/test/time1/03414.png",
        "industrial area": "0.6567",
        "commercial area": "0.2455",
        "dense residential": "0.0332",
        "building": "0.0115",
        "campus": "0.0078",
        "parking lot": "0.0060",
        "square": "0.0047",
        "stadium": "0.0042",
        "intersection": "0.0036",
        "basketball court": "0.0034",
        "single-family residential": "0.0032",
        "solar panel": "0.0027",
        "terrace": "0.0026",
        "interchange": "0.0019",
        "tennis court": "0.0014",
        "ground track field": "0.0012",
        "park": "0.0012",
        "container": "0.0011",
        "road": "0.0009",
        "storage tanks": "0.0009",
        "railway": "0.0006",
        "church": "0.0006",
        "harbor": "0.0005",
        "freeway": "0.0004",
        "runway": "0.0004",
        "fertile land": "0.0004",
        "oil tank": "0.0004",
        "airport": "0.0004",
        "island": "0.0003",
        "mine": "0.0003",
        "impermeable surface": "0.0003",
        "bare land": "0.0003",
        "farmland": "0.0002",
        "highway": "0.0002",
        "bridge": "0.0002",
        "ship": "0.0001",
        "sea": "0.0001",
        "pond": "0.0001",
        "river": "0.0001",
        "wetland": "0.0001",
        "cars": "0.0001",
        "shrubbery": "0.0000",
        "cotton field": "0.0000",
        "forest": "0.0000",
        "mountain": "0.0000",
        "golf course": "0.0000",
        "tree": "0.0000",
        "airplane": "0.0000",
        "snow land": "0.0000",
        "cabin": "0.0000",
        "prairie": "0.0000",
        "meadow": "0.0000",
        "lake": "0.0000",
        "chaparral": "0.0000",
        "beach": "0.0000",
        "desert": "0.0000"
    }
    jsonB = {
        "image_path": "/home/wangshiying/gjx/rschange/data/SYSU-CD/test/time2/03414.png",
        "industrial area": "0.5088",
        "commercial area": "0.2598",
        "dense residential": "0.0528",
        "interchange": "0.0253",
        "bridge": "0.0224",
        "campus": "0.0101",
        "building": "0.0096",
        "single-family residential": "0.0085",
        "river": "0.0078",
        "square": "0.0078",
        "harbor": "0.0070",
        "parking lot": "0.0058",
        "storage tanks": "0.0057",
        "freeway": "0.0048",
        "intersection": "0.0045",
        "stadium": "0.0043",
        "container": "0.0041",
        "basketball court": "0.0039",
        "railway": "0.0038",
        "ground track field": "0.0034",
        "tennis court": "0.0031",
        "solar panel": "0.0029",
        "park": "0.0028",
        "farmland": "0.0027",
        "road": "0.0027",
        "airport": "0.0027",
        "island": "0.0024",
        "pond": "0.0022",
        "highway": "0.0021",
        "terrace": "0.0020",
        "wetland": "0.0020",
        "fertile land": "0.0018",
        "runway": "0.0017",
        "ship": "0.0014",
        "church": "0.0010",
        "bare land": "0.0009",
        "oil tank": "0.0009",
        "impermeable surface": "0.0008",
        "mine": "0.0007",
        "cars": "0.0006",
        "sea": "0.0005",
        "airplane": "0.0004",
        "golf course": "0.0003",
        "shrubbery": "0.0002",
        "lake": "0.0002",
        "mountain": "0.0002",
        "cotton field": "0.0001",
        "chaparral": "0.0001",
        "tree": "0.0001",
        "forest": "0.0001",
        "snow land": "0.0001",
        "cabin": "0.0001",
        "prairie": "0.0000",
        "beach": "0.0000",
        "meadow": "0.0000",
        "desert": "0.0000"
    }



    # The following code is fixed, so you can modify it to make batch predictions for folder images.
    mean=[
        123.675,
        116.28,
        103.53,
        123.675,
        116.28,
        103.53,
    ]
    std=[
        58.395,
        57.12,
        57.375,
        58.395,
        57.12,
        57.375,
    ]
    preprocess_func = SegDataPreProcessor(mean=mean, std=std)
    device = torch.device(args.device)
    data, _ = _preprare_data([imgA_path, imgB_path], model)

    processed_data = preprocess_func(data)
    input_data = processed_data['inputs'].to(device)
    batch_img_metas = processed_data['data_samples']

    batch_img_metas[0].jsonA = list(jsonA.keys())[1:10]
    batch_img_metas[0].jsonB = list(jsonB.keys())[1:10]
    with torch.no_grad():
        results = model.whole_inference(input_data, batch_img_metas)
    
    # show the results
    results = torch.argmax(results, dim=1)
    pred_result = results[0].cpu().detach().numpy()
    pred_result = np.where(pred_result>0, 255, 0).astype(np.uint8)
    cv2.imwrite('000.png', pred_result)


if __name__ == '__main__':
    main()
