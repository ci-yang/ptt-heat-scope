#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTT 熱門文章預測 BERT 深度學習模型訓練腳本
使用中文BERT + 傳統特徵的混合架構
"""

import pickle
import re
import warnings

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModel,
    AutoTokenizer,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

warnings.filterwarnings("ignore")


class PTTBertDataset(Dataset):
    """PTT BERT數據集類"""

    def __init__(self, texts, numerical_features, labels, tokenizer, max_length=512):
        self.texts = texts
        self.numerical_features = torch.FloatTensor(numerical_features)
        self.labels = torch.LongTensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "numerical_features": self.numerical_features[idx],
            "labels": self.labels[idx],
        }


class PTTBertClassifier(nn.Module):
    """BERT + 數值特徵混合分類器"""

    def __init__(
        self, bert_model_name, n_numerical_features, n_classes=2, dropout_rate=0.3
    ):
        super(PTTBertClassifier, self).__init__()

        # BERT編碼器
        self.bert = AutoModel.from_pretrained(bert_model_name)

        # 凍結BERT部分參數（可選）
        # for param in self.bert.parameters():
        #     param.requires_grad = False

        # 特徵融合層
        bert_hidden_size = self.bert.config.hidden_size  # 通常是768

        # 數值特徵處理
        self.numerical_fc = nn.Sequential(
            nn.Linear(n_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # 特徵融合和分類
        combined_size = bert_hidden_size + 32  # BERT特徵 + 數值特徵
        self.classifier = nn.Sequential(
            nn.Linear(combined_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes),
        )

    def forward(self, input_ids, attention_mask, numerical_features):
        # BERT編碼
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # 使用[CLS] token的表示
        bert_features = bert_output.last_hidden_state[
            :, 0, :
        ]  # [batch_size, hidden_size]

        # 數值特徵處理
        numerical_features = self.numerical_fc(numerical_features)

        # 特徵融合
        combined_features = torch.cat([bert_features, numerical_features], dim=1)

        # 分類
        output = self.classifier(combined_features)

        return output


class PTTBertHotPostPredictor:
    def __init__(self, bert_model_name="bert-base-chinese"):
        """
        初始化BERT預測器

        Args:
            bert_model_name: BERT模型名稱
                - 'bert-base-chinese': Google中文BERT
                - 'hfl/chinese-bert-wwm-ext': 哈工大中文BERT
                - 'hfl/chinese-roberta-wwm-ext': 中文RoBERTa
        """
        self.bert_model_name = bert_model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_trained = False
        self.max_length = 512

        print(f"使用設備: {self.device}")
        print(f"BERT模型: {bert_model_name}")

    def preprocess_text(self, text):
        """文本預處理（BERT版本）"""
        if pd.isna(text) or text == "":
            return ""

        # 移除特殊字符和網址
        text = re.sub(r"http[s]?://\S+", "", text)
        text = re.sub(r"※.*", "", text)  # 移除PTT系統訊息
        text = re.sub(r"\s+", " ", text)  # 統一空白字符
        text = text.strip()

        return text

    def extract_numerical_features(self, df):
        """提取數值特徵"""
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

        # 文章類型特徵
        features["is_question"] = (
            df["title"].str.contains("問卦|請問", na=False).astype(int)
        )
        features["is_news"] = df["title"].str.contains("新聞", na=False).astype(int)
        features["has_bracket"] = (
            df["title"].str.contains(r"\[.*\]", na=False).astype(int)
        )

        return features

    def combine_text(self, title, content):
        """組合標題和內容為BERT輸入"""
        title = self.preprocess_text(title)
        content = self.preprocess_text(content)

        # 組合格式：[標題] [SEP] [內容]
        if content:
            combined = f"{title} [SEP] {content}"
        else:
            combined = title

        return combined

    def train(
        self, csv_path, epochs=10, batch_size=8, learning_rate=2e-5, warmup_steps=100
    ):
        """訓練BERT模型"""
        print("載入數據...")
        df = pd.read_csv(csv_path)

        # 過濾有效數據
        df = df.dropna(subset=["title", "is_hot"])
        df["content"] = df["content"].fillna("")

        print(f"總數據量: {len(df)}")
        hot_count = (df["is_hot"] == True).sum()
        print(f"熱門文章數: {hot_count}")
        print(f"普通文章數: {len(df) - hot_count}")

        # 初始化tokenizer
        print("載入BERT tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.bert_model_name)
        except Exception as e:
            print(f"載入 {self.bert_model_name} 失敗，改用 bert-base-chinese")
            self.bert_model_name = "bert-base-chinese"
            self.tokenizer = BertTokenizer.from_pretrained(self.bert_model_name)

        # 文本預處理
        print("處理文本數據...")
        df["combined_text"] = df.apply(
            lambda row: self.combine_text(row["title"], row["content"]), axis=1
        )

        # 數值特徵提取
        print("提取數值特徵...")
        numerical_features = self.extract_numerical_features(df)

        # 準備數據
        texts = df["combined_text"].values
        X_numerical = numerical_features.values
        y = (df["is_hot"] == True).astype(int).values

        # 分割數據
        X_text_train, X_text_test, X_num_train, X_num_test, y_train, y_test = (
            train_test_split(
                texts,
                X_numerical,
                y,
                test_size=0.2,
                random_state=42,
                stratify=y if len(set(y)) > 1 else None,  # type: ignore
            )
        )

        print(f"訓練集大小: {len(X_text_train)}")
        print(f"測試集大小: {len(X_text_test)}")
        print(f"數值特徵維度: {X_numerical.shape[1]}")

        # 創建數據集
        train_dataset = PTTBertDataset(
            X_text_train, X_num_train, y_train, self.tokenizer, self.max_length
        )
        test_dataset = PTTBertDataset(
            X_text_test, X_num_test, y_test, self.tokenizer, self.max_length
        )

        # 創建數據加載器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # 初始化模型
        print("初始化BERT分類器...")
        n_numerical_features = X_numerical.shape[1]
        self.model = PTTBertClassifier(
            self.bert_model_name, n_numerical_features, n_classes=2
        )
        self.model.to(self.device)

        # 設置優化器和調度器
        optimizer = optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=0.01
        )

        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )

        criterion = nn.CrossEntropyLoss()

        # 開始訓練
        print("開始訓練BERT模型...")
        best_accuracy = 0

        for epoch in range(epochs):
            # 訓練階段
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                input_ids = batch["input_ids"].to(self.device)  # type: ignore
                attention_mask = batch["attention_mask"].to(self.device)  # type: ignore
                numerical_features = batch["numerical_features"].to(self.device)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    numerical_features=numerical_features,
                )

                loss = criterion(outputs, labels)
                loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

                optimizer.step()
                scheduler.step()

                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()

            # 評估階段
            self.model.eval()
            test_loss = 0
            test_correct = 0
            test_total = 0
            all_predictions = []
            all_labels = []
            all_probabilities = []

            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch["input_ids"].to(self.device)  # type: ignore
                    attention_mask = batch["attention_mask"].to(self.device)  # type: ignore
                    numerical_features = batch["numerical_features"].to(self.device)
                    labels = batch["labels"].to(self.device)

                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        numerical_features=numerical_features,
                    )

                    loss = criterion(outputs, labels)
                    test_loss += loss.item()

                    probabilities = torch.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs.data, 1)

                    test_total += labels.size(0)
                    test_correct += (predicted == labels).sum().item()

                    all_predictions.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_probabilities.extend(probabilities[:, 1].cpu().numpy())

            train_accuracy = train_correct / train_total
            test_accuracy = test_correct / test_total

            if test_accuracy > best_accuracy:
                best_accuracy = test_accuracy

            # 顯示結果
            if len(set(all_labels)) > 1:
                auc_score = roc_auc_score(all_labels, all_probabilities)
            else:
                auc_score = 0.5

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print(
                f"訓練損失: {train_loss/len(train_loader):.4f}, 訓練準確率: {train_accuracy:.3f}"
            )
            print(
                f"測試損失: {test_loss/len(test_loader):.4f}, 測試準確率: {test_accuracy:.3f}"
            )
            print(f"AUC Score: {auc_score:.3f}")
            print(f"最佳準確率: {best_accuracy:.3f}")

        # 最終評估
        print("\n最終評估結果:")
        print(f"最佳測試準確率: {best_accuracy:.3f}")
        if len(set(all_labels)) > 1:
            final_auc = roc_auc_score(all_labels, all_probabilities)
            print(f"最終 AUC Score: {final_auc:.3f}")
            print("\n分類報告:")
            print(
                classification_report(
                    all_labels, all_predictions, target_names=["普通文章", "熱門文章"]
                )
            )

        self.is_trained = True
        print("\nBERT模型訓練完成！")

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

        # 文本處理
        combined_text = self.combine_text(title, content)

        # 數值特徵
        numerical_features = self.extract_numerical_features(temp_df)

        # Tokenization
        if self.tokenizer is None:
            raise ValueError("Tokenizer未初始化")

        encoding = self.tokenizer(
            combined_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)  # type: ignore
        attention_mask = encoding["attention_mask"].to(self.device)  # type: ignore
        numerical_tensor = torch.FloatTensor(numerical_features.values).to(self.device)

        # 預測
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                numerical_features=numerical_tensor,
            )
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

        # 保存模型狀態
        model_data = {
            "bert_model_name": self.bert_model_name,
            "model_state_dict": self.model.state_dict(),
            "max_length": self.max_length,
            "is_trained": self.is_trained,
        }

        with open(path, "wb") as f:
            pickle.dump(model_data, f)

        # 保存tokenizer
        tokenizer_path = path.replace(".pkl", "_tokenizer")
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(tokenizer_path)

        print(f"BERT模型已保存到: {path}")
        print(f"Tokenizer已保存到: {tokenizer_path}")

    def load_model(self, path):
        """載入模型"""
        with open(path, "rb") as f:
            model_data = pickle.load(f)

        self.bert_model_name = model_data["bert_model_name"]
        self.max_length = model_data["max_length"]
        self.is_trained = model_data["is_trained"]

        # 載入tokenizer
        tokenizer_path = path.replace(".pkl", "_tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # 重建模型（需要知道數值特徵維度）
        n_numerical_features = 12  # 根據extract_numerical_features的特徵數量
        self.model = PTTBertClassifier(
            self.bert_model_name, n_numerical_features, n_classes=2
        )
        self.model.load_state_dict(model_data["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"BERT模型已載入: {path}")


def main():
    """主函數"""
    # 可選的中文BERT模型
    bert_models = [
        "bert-base-chinese",  # Google中文BERT
        "hfl/chinese-bert-wwm-ext",  # 哈工大中文BERT
        "hfl/chinese-roberta-wwm-ext",  # 中文RoBERTa
    ]

    predictor = PTTBertHotPostPredictor(bert_model_name=bert_models[0])

    # 訓練模型
    csv_path = "data/raw/ptt_data_20250616_203938_0000.csv"
    predictor.train(
        csv_path,
        epochs=5,  # BERT通常需要較少的epoch
        batch_size=4,  # 較小的batch size避免記憶體不足
        learning_rate=2e-5,  # BERT建議的學習率
        warmup_steps=50,
    )

    # 保存模型
    predictor.save_model("models/ptt_bert_predictor.pkl")

    # 測試預測
    print("\n測試BERT預測:")
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
