import torch
import torch.nn.functional as F
from fastapi import APIRouter
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Tạo Router riêng cho Toxic
router = APIRouter(prefix="/toxic", tags=["Toxic Detection"])

# Cấu hình
MODEL_PATH = "./model"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = None
model = None

class CommentRequest(BaseModel):
    text: str

# Hàm load model
def load_model():
    global tokenizer, model
    print(f"[Toxic] Đang tải model lên {device}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("[Toxic] Model đã sẵn sàng!")
    except Exception as e:
        print(f"[Toxic] Lỗi tải model: {e}")

@router.post("/predict")
async def predict_toxic(req: CommentRequest):
    if not model or not tokenizer:
        return {"error": "Toxic model chưa tải xong hoặc bị lỗi."}

    # 1. Tokenize
    inputs = tokenizer(
        req.text, 
        return_tensors="pt", 
        truncation=True, 
        max_length=128, 
        padding=True
    ).to(device)

    # 2. Predict
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. Softmax & Score
    probs = F.softmax(outputs.logits, dim=1)
    toxic_score = probs[0][1].item()
    is_toxic = toxic_score > 0.5 

    return {
        "is_toxic": is_toxic,
        "confidence": round(toxic_score, 4),
        "label": "TOXIC" if is_toxic else "SAFE"
    }