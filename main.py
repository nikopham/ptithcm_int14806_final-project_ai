from fastapi import FastAPI
import uvicorn

# Import 2 module con
import toxic_service
import recommend_service

# Khởi tạo App
app = FastAPI(title="Movie AI System (Toxic Check + Recommender)")

app.include_router(toxic_service.router)
app.include_router(recommend_service.router)

@app.on_event("startup")
async def startup_event():
    print("\nĐang khởi động Server...")

    toxic_service.load_model()
    recommend_service.load_model()
    
    print("Server đã sẵn sàng!\n")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)