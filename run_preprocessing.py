import os
import sys
import numpy as np
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processing.preprocessor import CWNUDataPreprocessor
from src.utils.config_loader import load_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config = load_config()
    data_config = config.get('data', {})
    
    logger.info("CWRU轴承故障数据预处理")
    
    preprocessor = CWNUDataPreprocessor(data_config)
    data_dir = data_config.get('data_dir', 'data')
    
    preprocessor.segment_length = 1536
    logger.info(f"信号长度: {preprocessor.segment_length}")
    
    logger.info(f"数据目录: {os.path.abspath(data_dir)}")
    logger.info("开始处理...")
    
    X_train, X_val, X_test, y_train, y_val, y_test = preprocessor.prepare_dataset(data_dir)
    
    logger.info("预处理完成")
    logger.info(f"训练集: {X_train.shape}")
    logger.info(f"验证集: {X_val.shape}")
    logger.info(f"测试集: {X_test.shape}")
    
    #保存数据集
    processed_dir = data_config.get('processed_dir', 'data/processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    save_path = os.path.join(processed_dir, 'dataset.npz')
    np.savez(save_path,
             X_train=X_train, y_train=y_train,
             X_val=X_val, y_val=y_val,
             X_test=X_test, y_test=y_test)
    logger.info(f"数据集保存: {os.path.abspath(save_path)}")
    
    #保存预处理器
    try:
        import joblib
        preprocessor_path = os.path.join(processed_dir, 'preprocessor.pkl')
        joblib.dump(preprocessor, preprocessor_path)
        logger.info(f"预处理器保存: {os.path.abspath(preprocessor_path)}")
    except Exception as e:
        logger.warning(f"joblib保存失败: {e}")
    
    #保存板端归一化参数
    preprocessor.save_board_params(processed_dir)
    logger.info(f"板端参数保存: {os.path.join(processed_dir, 'board_norm.json')}")
    
    logger.info("全部完成")


if __name__ == "__main__":
    main()