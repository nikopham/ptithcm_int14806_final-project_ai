import pandas as pd
from surprise import KNNBasic, Dataset, Reader # <-- Dùng KNNBasic thay vì SVD
import pickle
import os
from typing import List
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter(prefix="/movie", tags=["Movie Recommender"])

MODEL_FILE = "knn_model.pkl"
algo = None

class RatingData(BaseModel):
    userId: str
    movieId: str
    rating: float

class TrainRequest(BaseModel):
    data: List[RatingData]

class RecommendRequest(BaseModel):
    userId: str
    allMovieIds: List[str]

def load_model():
    global algo
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, 'rb') as f:
                algo = pickle.load(f)
            print("[Movie-KNN] Model loaded successfully!")
        except Exception as e:
            print(f"[Movie-KNN] Lỗi đọc file model: {e}")
    else:
        print("[Movie-KNN] Chưa có file model. Vui lòng gọi API /train.")

@router.post("/train")
async def train_knn(req: TrainRequest):
    global algo

    if not req.data:
        return {"status": "error", "message": "No data sent"}

    print(f"[Movie-KNN] Bắt đầu train với {len(req.data)} bản ghi...")

    # b. Chuyển JSON -> DataFrame
    # Surprise yêu cầu đúng thứ tự cột: user -> item -> rating
    df = pd.DataFrame([d.dict() for d in req.data])
    df = df[['userId', 'movieId', 'rating']]

    # c. Cấu hình Reader & Dataset
    reader = Reader(rating_scale=(0, 5)) # Thang điểm 0-5
    data = Dataset.load_from_df(df, reader)
    trainset = data.build_full_trainset()

    # d. CẤU HÌNH THUẬT TOÁN KNN
    sim_options = {
        'name': 'cosine',   # Cách tính khoảng cách (Cosine Similarity)
        'user_based': True, # True = Tìm người giống người (User-based)

        'min_support': 1    # Số lượng item chung tối thiểu để coi là có liên quan
    }

    new_algo = KNNBasic(sim_options=sim_options)

    # e. Bắt đầu học (Tính toán ma trận tương đồng)
    new_algo.fit(trainset)

    # f. Lưu model
    algo = new_algo
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(algo, f)

    return {"status": "success", "message": "Model KNN đã train xong (User-based)"}

# --- 4. API RECOMMEND ---
@router.post("/recommend")
async def recommend_movie(req: RecommendRequest):
    # Kiểm tra model đã sẵn sàng chưa
    if not algo:
        return {"movieIds": [], "message": "Model chưa train/load"}

    predictions = []

    # Duyệt qua từng phim trong danh sách ứng viên
    for mid in req.allMovieIds:
        # Dự đoán: User này sẽ chấm phim 'mid' bao nhiêu điểm?
        # Logic KNN: Tìm k người giống User nhất đã xem phim 'mid', lấy điểm trung bình của họ
        pred = algo.predict(req.userId, mid)
        predictions.append((mid, pred.est))

        # Sắp xếp điểm từ cao xuống thấp
    predictions.sort(key=lambda x: x[1], reverse=True)

    # Lấy Top 20 ID phim
    top_ids = [mid for mid, score in predictions[:20]]

    return {"movieIds": top_ids, "debug_score": predictions[:5]}