from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from google.cloud import storage
import joblib
import os

app = FastAPI()

# Đọc cấu hình từ biến môi trường
GCS_BUCKET = os.environ.get("GCS_BUCKET", "mlops-lab-vinuni-21-storage")
GCS_MODEL_KEY = "models/latest/model.pkl"
# Đường dẫn model (Ưu tiên thư mục cục bộ nếu có, nếu không dùng đường dẫn VM)
LOCAL_MODEL_PATH = "models/model.pkl"
VM_MODEL_PATH = os.path.expanduser("~/models/model.pkl")
MODEL_PATH = LOCAL_MODEL_PATH if os.path.exists(LOCAL_MODEL_PATH) else VM_MODEL_PATH

def download_model():
    """Tải file model.pkl từ GCS về máy khi server khởi động."""
    if not os.path.exists(os.path.dirname(MODEL_PATH)):
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    
    try:
        client = storage.Client()
        bucket = client.bucket(GCS_BUCKET)
        blob = bucket.blob(GCS_MODEL_KEY)
        blob.download_to_filename(MODEL_PATH)
        print(f"Successfully downloaded model from gs://{GCS_BUCKET}/{GCS_MODEL_KEY}")
    except Exception as e:
        print(f"Warning: Could not download model from GCS: {e}")
        # Trong thực tế, nếu không có model thì server không nên chạy, 
        # nhưng ở đây ta để nó tiếp tục để có thể debug hoặc dùng model cũ nếu có.

# Tải model khi khởi động
if not os.environ.get("SKIP_MODEL_DOWNLOAD"):
    download_model()

# Load model vào bộ nhớ
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None
    print("Warning: Model file not found at", MODEL_PATH)

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    """Endpoint kiểm tra sức khỏe server."""
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luận.
    Đầu vào: JSON {"features": [f1, f2, ..., f12]}
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    if len(req.features) != 12:
        raise HTTPException(
            status_code=400, 
            detail=f"Expected 12 features (wine quality), but got {len(req.features)}"
        )

    try:
        prediction = int(model.predict([req.features])[0])
        labels = {0: "thấp", 1: "trung bình", 2: "cao"}
        return {
            "prediction": prediction,
            "label": labels.get(prediction, "không xác định")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
