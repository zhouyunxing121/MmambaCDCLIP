import os
import torch
import torch.nn.functional as F
from dassl.engine import build_trainer
from dassl.config import get_cfg_default

# ⚠️ 注意：这里需要导入你在 Dassl/ProText 中注册的 trainer
# 假设你在 trainers 文件夹下定义了 ProTextTrainer
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../ProText')) # 确保能找到 ProText 目录
from trainers.protext_forcd import ProText  # 或者是 protext.py，取决于你实际运行成功的名字

def setup_cfg(config_file, model_dir):
    cfg = get_cfg_default()
    # 合并你的 yaml 配置文件
    cfg.merge_from_file(config_file)
    cfg.MODEL.BACKBONE.NAME = "ViT-L/14" # 必须与你训练时保持绝对一致！
    cfg.DATASET.NAME = "LevirCD"
    cfg.OUTPUT_DIR = model_dir
    cfg.freeze()
    return cfg

def main():
    # 1. 配置路径
    # ⚠️ 替换为你实际训练 ProText 使用的 config 文件路径
    config_file = "ProText/configs/trainers/ProText/text_only_supervised/levir_cd.yaml" 
    # ⚠️ 替换为你训练好的 ProText 权重文件夹
    model_dir = "ProText/output/levir_cd/ProText/RN50/seed1" 

    cfg = setup_cfg(config_file, model_dir)

    # 2. 构建 Trainer (它会自动加载模型结构)
    trainer = build_trainer(cfg)
    
    # 3. 加载训练好的权重 (假设跑了 50 个 epoch)
    trainer.load_model(model_dir, epoch=50) 
    
    # 提取核心的 prompt_learner 和 text_encoder
    prompt_learner = trainer.model.prompt_learner
    text_encoder = trainer.model.text_encoder
    
    prompt_learner.eval()
    text_encoder.eval()

    # 4. 类别名称 (严格保持与训练一致)
    classnames = ["unchanged", "changed"]
    
    print("开始提取 ProText 专家文本特征...")
    with torch.no_grad():
        # 获取 prompt (结合了学到的 ctx 和 classnames)
        prompts = prompt_learner(classnames)
        # 获取文本编码器的 Tokenized IDs
        tokenized_prompts = prompt_learner.tokenized_prompts
        
        # 输入 text_encoder 获取特征
        text_features = text_encoder(prompts, tokenized_prompts) # 预期形状: [2, 特征维度(如 1024 或 768)]
        
        # ⚠️ 极其重要：L2 归一化
        text_features = F.normalize(text_features, dim=-1)

    # 5. 保存特征
    save_path = "/home/dc001/clip3-2/protext_expert_features_levir.pt"
    torch.save(text_features.cpu(), save_path)
    print(f"✅ ProText 专家特征提取成功！已保存至: {save_path}")
    print(f"特征形状: {text_features.shape}")

if __name__ == "__main__":
    main()