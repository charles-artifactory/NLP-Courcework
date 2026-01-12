#!/usr/bin/env python3
"""
RAG增强智能问答系统 - 主入口

使用方法:
    # 启动Web界面
    python main.py
    
    # 启动API服务
    python main.py --api
    
    # 指定端口
    python main.py --port 8080
    
    # 创建公共链接
    python main.py --share
"""

import argparse
import logging
import sys
from pathlib import Path

# 添加项目根目录到路径
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))


def setup_logging(level: str = "INFO"):
    """配置日志系统"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(ROOT_DIR / "rag_system.log")
        ]
    )
    
    # 降低第三方库的日志级别
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="RAG增强智能问答系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py                  # 启动Gradio Web界面
    python main.py --api            # 启动FastAPI服务
    python main.py --port 8080      # 指定端口
    python main.py --share          # 创建公共分享链接
        """
    )
    
    parser.add_argument(
        "--api",
        action="store_true",
        help="启动FastAPI服务（默认启动Gradio界面）"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="服务地址 (默认: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="服务端口 (默认: 7860)"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="创建Gradio公共分享链接"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="日志级别 (默认: INFO)"
    )
    
    args = parser.parse_args()
    
    # 配置日志
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 50)
    logger.info("RAG增强智能问答系统")
    logger.info("=" * 50)
    
    if args.api:
        # 启动API服务
        logger.info(f"启动FastAPI服务: http://{args.host}:{args.port}")
        logger.info("API文档: http://{args.host}:{args.port}/docs")
        
        from src.api import run_api
        run_api(host=args.host, port=args.port)
    else:
        # 启动Gradio界面
        logger.info(f"启动Gradio界面: http://{args.host}:{args.port}")
        
        from app.gradio_app import launch_app
        launch_app(host=args.host, port=args.port, share=args.share)


if __name__ == "__main__":
    main()
