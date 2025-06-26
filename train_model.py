#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTT 熱門文章預測深度學習模型訓練腳本
"""

import pickle
import re
import warnings

import jieba
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")


class PTTDataset(Dataset):
    """PTT數據集類"""

    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PTTClassifier(nn.Module):
    """深度學習分類器"""

    def __init__(self, input_dim, hidden_dims=[512, 256, 128], dropout_rate=0.5):
        super(PTTClassifier, self).__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # 輸出層
        layers.append(nn.Linear(prev_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PTTHotPostPredictor:
    def __init__(self):
        self.tfidf_title = TfidfVectorizer(max_features=1000, stop_words=None)
        self.tfidf_content = TfidfVectorizer(max_features=2000, stop_words=None)
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False

        print(f"使用設備: {self.device}")

    def preprocess_text(self, text):
        """文本預處理"""
        if pd.isna(text) or text == "":
            return ""

        # 移除特殊字符和網址
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"※.*", "", text)  # 移除PTT系統訊息
        text = re.sub(r"[^\w\s]", " ", text)

        # 使用jieba分詞
        words = jieba.cut(text)
        return " ".join(words)

    def extract_features(self, df):
        """特徵工程"""
        features = pd.DataFrame()

        # 文字長度特徵
        features["title_length"] = df["title"].str.len().fillna(0)
        features["content_length"] = df["content"].str.len().fillna(0)

        # 數值特徵
        features["reply_count"] = pd.to_numeric(
            df["reply_count"], errors="coerce"
        ).fillna(0)
        features["push_count"] = pd.to_numeric(
            df["push_count"], errors="coerce"
        ).fillna(0)
        features["boo_count"] = pd.to_numeric(df["boo_count"], errors="coerce").fillna(
            0
        )

        # 衍生特徵
        features["push_ratio"] = features["push_count"] / (
            features["push_count"] + features["boo_count"] + 1
        )
        features["engagement"] = (
            features["reply_count"] + features["push_count"] + features["boo_count"]
        )

        # 時間特徵
        df["post_time"] = pd.to_datetime(df["post_time"], errors="coerce")
        features["hour"] = df["post_time"].dt.hour.fillna(12)
        features["day_of_week"] = df["post_time"].dt.dayofweek.fillna(0)

        # 文章類型特徵（根據標題判斷）
        features["is_question"] = (
            df["title"].str.contains("問卦|請問", na=False).astype(int)
        )
        features["is_news"] = df["title"].str.contains("新聞", na=False).astype(int)
        features["has_bracket"] = (
            df["title"].str.contains(r"\[.*\]", na=False).astype(int)
        )

        return features

    def train(self, csv_path, epochs=50, batch_size=8, learning_rate=0.001):
        """訓練深度學習模型"""
        print("載入數據...")
        df = pd.read_csv(csv_path)

        # 過濾有效數據
        df = df.dropna(subset=["title", "is_hot"])
        df["content"] = df["content"].fillna("")

        print(f"總數據量: {len(df)}")
        hot_count = (df["is_hot"] == True).sum()
        print(f"熱門文章數: {hot_count}")
        print(f"普通文章數: {len(df) - hot_count}")

        # 文本預處理
        print("處理文本數據...")
        df["processed_title"] = df["title"].apply(self.preprocess_text)
        df["processed_content"] = df["content"].apply(self.preprocess_text)

        # 特徵工程
        print("提取特徵...")
        numerical_features = self.extract_features(df)

        # TF-IDF特徵
        title_tfidf = self.tfidf_title.fit_transform(df["processed_title"])
        content_tfidf = self.tfidf_content.fit_transform(df["processed_content"])

        # 合併所有特徵
        from scipy.sparse import hstack

        X_text = hstack([title_tfidf, content_tfidf])
        X_numerical = numerical_features.values

        # 將稀疏矩陣轉換為密集矩陣並合併
        X_text_dense = X_text.toarray()
        X = np.concatenate([X_text_dense, X_numerical], axis=1)

        # 目標變量
        y = (df["is_hot"] == True).astype(int).values

        # 分割數據
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"訓練集大小: {len(X_train)}")
        print(f"測試集大小: {len(X_test)}")
        print(f"特徵維度: {X.shape[1]}")

        # 創建數據集
        train_dataset = PTTDataset(X_train, y_train)
        test_dataset = PTTDataset(X_test, y_test)

        # 創建數據加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        print("初始化深度神經網絡...")
        input_dim = X.shape[1]
        self.model = PTTClassifier(
            input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3
        )
        self.model.to(self.device)

        # 設置優化器和損失函數
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

        # 開始訓練
        print("開始訓練...")
        best_accuracy = 0

        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0
            train_correct = 0

            for features, labels in train_loader:
                features, labels = features.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(features)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_correct += (predicted == labels).sum().item()

            # 評估階段
            self.model.eval()
            test_loss = 0
            test_correct = 0
            all_predictions = []
            all_labels = []
            all_probabilities = []

            with torch.no_grad():
                for features, labels in test_loader:
                    features, labels = features.to(self.device), labels.to(self.device)
                    outputs = self.model(features)
                    loss = criterion(outputs, labels)

                    test_loss += loss.item()
                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)
                    test_correct += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())

            train_accuracy = train_correct / len(train_dataset)
            test_accuracy = test_correct / len(test_dataset)

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

            # 每10個epoch顯示一次詳細結果
            if (epoch + 1) % 10 == 0:
                auc_score = roc_auc_score(all_labels, all_probabilities)
                print(f"\nEpoch {epoch + 1}/{epochs}")
                print(
                    f"訓練損失: {train_loss/len(train_loader):.4f}, 訓練準確率: {train_accuracy:.3f}"
                )
                print(
                    f"測試損失: {test_loss/len(test_loader):.4f}, 測試準確率: {test_accuracy:.3f}"
                )
                print(f"AUC Score: {auc_score:.3f}")
                print(f"最佳準確率: {best_accuracy:.3f}")

            scheduler.step()

        # 最終評估
        print("\n最終評估結果:")
        print(f"最佳測試準確率: {best_accuracy:.3f}")
        if len(set(all_labels)) > 1:  # 確保有兩個類別
            final_auc = roc_auc_score(all_labels, all_probabilities)
            print(f"最終 AUC Score: {final_auc:.3f}")
            print("\n分類報告:")
            print(
                classification_report(
                    all_labels, all_predictions, target_names=["普通文章", "熱門文章"]
                )
            )

        self.is_trained = True
        print("\n深度學習模型訓練完成！")

    def predict(self, title, content=""):
        """預測單篇文章"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未訓練！")

        # 創建臨時DataFrame
        temp_df = pd.DataFrame(
            {
                "title": [title],
                "content": [content],
                "reply_count": [0],
                "push_count": [0],
                "boo_count": [0],
                "post_time": [pd.Timestamp.now()],
            }
        )

        # 預處理
        temp_df["processed_title"] = temp_df["title"].apply(self.preprocess_text)
        temp_df["processed_content"] = temp_df["content"].apply(self.preprocess_text)

        # 特徵提取
        numerical_features = self.extract_features(temp_df)
        title_tfidf = self.tfidf_title.transform(temp_df["processed_title"])
        content_tfidf = self.tfidf_content.transform(temp_df["processed_content"])

        # 合併特徵
        from scipy.sparse import hstack

        X_text = hstack([title_tfidf, content_tfidf])
        X_text_dense = X_text.toarray()
        X = np.concatenate([X_text_dense, numerical_features.values], axis=1)

        # 轉換為tensor
        X_tensor = torch.FloatTensor(X).to(self.device)

        # 預測
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            prediction = torch.argmax(outputs, dim=1)

        hot_probability = probabilities[0][1].item()
        is_hot = bool(prediction[0].item())

        return {
            "is_hot": is_hot,
            "hot_probability": hot_probability,
            "confidence_level": self._get_confidence_level(hot_probability),
        }

    def _get_confidence_level(self, probability):
        """根據機率返回信心等級"""
        if probability >= 0.8:
            return "非常高"
        elif probability >= 0.6:
            return "高"
        elif probability >= 0.4:
            return "中等"
        elif probability >= 0.2:
            return "低"
        else:
            return "非常低"

    def save_model(self, path):
        """保存模型"""
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未訓練，無法保存！")

        model_data = {
            "tfidf_title": self.tfidf_title,
            "tfidf_content": self.tfidf_content,
            "model_state_dict": self.model.state_dict(),
            "model_input_dim": list(self.model.network.children())[0].in_features,
            "is_trained": self.is_trained,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        print(f"深度學習模型已保存到: {path}")

    def load_model(self, path):
        """載入模型"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.tfidf_title = model_data["tfidf_title"]
        self.tfidf_content = model_data["tfidf_content"]
        self.is_trained = model_data["is_trained"]

        # 重建模型
        input_dim = model_data["model_input_dim"]
        self.model = PTTClassifier(
            input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3
        )
        self.model.load_state_dict(model_data["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"深度學習模型已載入: {path}")


def main():
    """主函數"""
    predictor = PTTHotPostPredictor()

    # 訓練模型
    csv_path = "data/raw/ptt_data_20250616_203938_0000.csv"
    predictor.train(csv_path, epochs=50, batch_size=4, learning_rate=0.001)

    # 保存模型
    predictor.save_model("models/ptt_hot_predictor.pkl")

    # 測試預測
    print("\n測試預測:")
    test_cases = [
        "[問卦] 為什麼台灣年輕人都不想生小孩？",
        "[新聞] 台積電股價再創新高",
        "[心得] 今天去吃了很好吃的拉麵",
        "[爆卦] 重大發現！外星人降落地球了",
        "[問卦] 大家覺得現在房價合理嗎？",
    ]

    for title in test_cases:
        result = predictor.predict(title)
        print(f"\n標題: {title}")
        print(f"預測結果: {'熱門' if result['is_hot'] else '普通'}")
        print(f"熱門機率: {result['hot_probability']:.2%}")
        print(f"信心等級: {result['confidence_level']}")


if __name__ == "__main__":
    main()
