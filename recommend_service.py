import pandas as pd
from surprise import SVD, Dataset, Reader
import pickle
import os
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/movie", tags=["Movie Recommender"])

MODEL_FILE = "svd_model.pkl"
algo = None

# --- DTO ---
class RatingData(BaseModel):
    userId: str
    movieId: str
    rating: float

class TrainRequest(BaseModel):
    data: List[RatingData]

class RecommendRequest(BaseModel):
    userId: str
    allMovieIds: List[str]

# Hàm load mode
def load_model():
    global algo
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                algo = pickle.load(f)
            print("[Movie] SVD Model loaded successfully!")
        except Exception as e:
            print(f"[Movie] Lỗi đọc file model: {e}")
    else:
        print("[Movie] Chưa có file model SVD. Vui lòng gọi API /train trước.")

@router.post("/train")
async def train_svd(req: TrainRequest):
    global algo
    if not req.data: 
        return {"status": "error", "message": "No data"}

    print(f"[Movie] Bắt đầu train SVD với {len(req.data)} bản ghi...")
    
    # Convert JSON -> DataFrame
    df = pd.DataFrame([d.dict() for d in req.data])
    df = df[['userId', 'movieId', 'rating']] # Thứ tự cột bắt buộc của Surprise
    
    # Cấu hình Surprise
    reader = Reader(rating_scale=(0, 5))
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()
    
    # Train
    new_algo = SVD()
    new_algo.fit(trainset)
    
    # Lưu Model
    algo = new_algo
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(algo, f)
        
    return {"status": "success", "message": "Model SVD đã train và lưu thành công"}

@router.post("/recommend")
async def recommend_movie(req: RecommendRequest):
    if not algo:
        return {"movieIds": [], "message": "Model chưa train"}

    predictions = []
    for mid in req.allMovieIds:
        # Dự đoán rating cho từng phim
        pred = algo.predict(req.userId, mid)
        predictions.append((mid, pred.est)) 

    # Sắp xếp giảm dần theo điểm dự đoán
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy Top 20 ID phim
    top_ids = [mid for mid, score in predictions[:20]]

    return {"movieIds": top_ids}