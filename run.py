#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTT 熱門文章預測系統運行腳本
"""

import argparse
import os
import subprocess
import sys


def install_dependencies():
    """安裝依賴包"""
    print("🔄 安裝依賴包...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    print("✅ 依賴包安裝完成")


def train_model():
    """訓練模型"""
    print("🤖 開始訓練模型...")
    subprocess.run([sys.executable, "train_model.py"])
    print("✅ 模型訓練完成")


def start_server():
    """啟動FastAPI服務"""
    print("🚀 啟動 PTT 熱門文章預測服務...")
    print("📱 服務地址: http://localhost:8000")
    subprocess.run([sys.executable, "app.py"])


def main():
    parser = argparse.ArgumentParser(description="PTT 熱門文章預測系統")
    parser.add_argument("--install", action="store_true", help="安裝依賴包")
    parser.add_argument("--train", action="store_true", help="只訓練模型")
    parser.add_argument("--serve", action="store_true", help="只啟動服務")
    parser.add_argument(
        "--all", action="store_true", help="完整流程：安裝依賴 -> 訓練模型 -> 啟動服務"
    )

    args = parser.parse_args()

    if args.install:
        install_dependencies()
    elif args.train:
        train_model()
    elif args.serve:
        start_server()
    elif args.all:
        install_dependencies()
        train_model()
        start_server()
    else:
        print("🔥 PTT 熱門文章預測系統")
        print("使用方式:")
        print("  python run.py --install     # 安裝依賴包")
        print("  python run.py --train       # 訓練模型")
        print("  python run.py --serve       # 啟動服務")
        print("  python run.py --all         # 完整流程")


if __name__ == "__main__":
    main()
