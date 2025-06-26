#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PTT 熱門文章預測 FastAPI 應用
"""

import logging
import os

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from train_model import PTTHotPostPredictor

# 設置日誌
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 創建FastAPI應用
app = FastAPI(
    title="PTT 熱門文章預測器",
    description="使用機器學習預測PTT文章是否會成為熱門文章",
    version="1.0.0",
)

# 設置模板和靜態文件
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# 全局變量存儲模型
predictor = None


class PredictionRequest(BaseModel):
    title: str
    content: str = ""


class PredictionResponse(BaseModel):
    title: str
    content: str
    is_hot: bool
    hot_probability: float
    confidence_level: str
    heat_score: int  # 1-100的熱度分數
    recommendation: str


@app.on_event("startup")
async def startup_event():
    """應用啟動時載入模型"""
    global predictor
    try:
        predictor = PTTHotPostPredictor()
        model_path = "models/ptt_hot_predictor.pkl"

        if os.path.exists(model_path):
            predictor.load_model(model_path)
            logger.info("模型載入成功")
        else:
            logger.warning(f"模型文件不存在: {model_path}")
            logger.info("請先運行 train_model.py 訓練模型")

    except Exception as e:
        logger.error(f"載入模型失敗: {e}")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """首頁"""
    return templates.TemplateResponse(
        "index.html", {"request": request, "title": "PTT 熱門文章預測器"}
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_article(request: PredictionRequest):
    """預測文章熱門程度API"""
    global predictor

    if predictor is None or not predictor.is_trained:
        raise HTTPException(status_code=503, detail="模型尚未載入，請稍後再試")

    if not request.title.strip():
        raise HTTPException(status_code=400, detail="文章標題不能為空")

    try:
        # 進行預測
        result = predictor.predict(request.title, request.content)

        # 計算熱度分數 (1-100)
        heat_score = int(result["hot_probability"] * 100)

        # 生成建議
        recommendation = _generate_recommendation(
            result["hot_probability"], request.title, request.content
        )

        return PredictionResponse(
            title=request.title,
            content=request.content,
            is_hot=result["is_hot"],
            hot_probability=result["hot_probability"],
            confidence_level=result["confidence_level"],
            heat_score=heat_score,
            recommendation=recommendation,
        )

    except Exception as e:
        logger.error(f"預測失敗: {e}")
        raise HTTPException(status_code=500, detail=f"預測過程中發生錯誤: {str(e)}")


@app.get("/api/health")
async def health_check():
    """健康檢查API"""
    global predictor

    return {
        "status": "healthy",
        "model_loaded": predictor is not None and predictor.is_trained,
        "version": "1.0.0",
    }


def _generate_recommendation(probability: float, title: str, content: str) -> str:
    """根據預測結果生成建議"""

    recommendations = []

    if probability >= 0.7:
        recommendations.append("🔥 您的文章很有潛力成為熱門！")
        recommendations.append("建議在人流量高的時段發文（晚上8-11點）")
    elif probability >= 0.5:
        recommendations.append("📈 文章有一定熱門潛力")
        recommendations.append("可以考慮在標題中加入更吸引人的關鍵字")
    elif probability >= 0.3:
        recommendations.append("💡 文章需要一些優化")
        recommendations.append("建議：1) 標題更具爆點 2) 內容更豐富有趣")
    else:
        recommendations.append("📝 建議重新思考文章方向")
        recommendations.append("可以參考熱門文章的寫作模式")

    # 根據標題特徵給建議
    if "[問卦]" in title:
        recommendations.append("💬 問卦文章建議提出有爭議性或有趣的問題")
    elif "[新聞]" in title:
        recommendations.append("📰 新聞文章建議選擇時事熱點話題")
    elif "[心得]" in title:
        recommendations.append("💭 心得文章建議分享獨特或有共鳴的經驗")

    # 內容長度建議
    if len(content) < 50:
        recommendations.append("📄 建議增加文章內容長度，更詳細的描述會增加吸引力")

    return " | ".join(recommendations)


@app.get("/api/stats")
async def get_stats():
    """獲取統計信息"""
    # 這裡可以添加更多統計信息
    return {
        "total_predictions": "功能開發中",
        "accuracy": "功能開發中",
        "popular_keywords": ["問卦", "台灣", "政治", "生活", "科技"],
    }


if __name__ == "__main__":
    # 確保必要的目錄存在
    os.makedirs("models", exist_ok=True)
    os.makedirs("templates", exist_ok=True)
    os.makedirs("static", exist_ok=True)

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
