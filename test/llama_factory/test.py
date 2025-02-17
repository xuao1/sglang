from llamafactory.train.tuner import run_exp
import os
import yaml
from pathlib import Path

def main():
    # 1. 加载 YAML 文件为字典
    config_path = Path(__file__).parent / "llama3_lora_sft.yaml"  # 确保路径正确
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)  # 关键步骤：将 YAML 转为字典
    
    # 2. 传递字典而非路径
    run_exp(args=config) 

if __name__ == "__main__":
    os.environ["FORCE_TORCHRUN"] = "0"
    main()
