# from fastapi import FastAPI
# import uvicorn
#
# # Import 2 module con
# import toxic_service
# import recommend_service
#
# # Khởi tạo App
# app = FastAPI(title="Movie AI System (Toxic Check + Recommender)")
#
# app.include_router(toxic_service.router)
# app.include_router(recommend_service.router)
#
# @app.on_event("startup")
# async def startup_event():
#     print("\nĐang khởi động Server...")
#
#     toxic_service.load_model()
#     recommend_service.load_model()
#
#     print("Server đã sẵn sàng!\n")
#
# if __name__ == "__main__":
#     uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=
import os
import json
import shutil
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager
from google.cloud import storage
from google.oauth2 import service_account
import base64

import toxic_service
import recommend_service

# --- CẤU HÌNH GCS ---
GCS_SOURCE_FOLDER = "model"
LOCAL_DESTINATION_FOLDER = "./model"
BUCKET_NAME = "comment-filter"


def download_from_gcs_folder():
    if os.path.exists(LOCAL_DESTINATION_FOLDER) and os.listdir(LOCAL_DESTINATION_FOLDER):
        print(f"Model đã có sẵn tại {LOCAL_DESTINATION_FOLDER}. Bỏ qua tải xuống.")
        return True

    print(f"Đang kết nối GCS để tải folder '{GCS_SOURCE_FOLDER}'...")

    b64_str = os.environ.get("GCP_CREDENTIALS_BASE64")

    if not b64_str:
        print("Lỗi: Không tìm thấy biến môi trường GCP_CREDENTIALS_JSON")
        return False

    try:
        # 2. Kết nối GCS
        if b64_str:
            json_str = base64.b64decode(b64_str).decode('utf-8')
        creds_dict = json.loads(json_str)
        credentials = service_account.Credentials.from_service_account_info(creds_dict)
        storage_client = storage.Client(credentials=credentials)
        bucket = storage_client.bucket(BUCKET_NAME)

        blobs = bucket.list_blobs(prefix=GCS_SOURCE_FOLDER)

        files_count = 0
        found_any = False

        for blob in blobs:
            found_any = True
            if blob.name.endswith("/"): continue

            # Xử lý đường dẫn file
            relative_path = os.path.relpath(blob.name, GCS_SOURCE_FOLDER)
            local_file_path = os.path.join(LOCAL_DESTINATION_FOLDER, relative_path)

            # Tạo thư mục và tải file
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            print(f"⬇Downloading: {relative_path}")
            blob.download_to_filename(local_file_path)
            files_count += 1

        if files_count > 0:
            print(f"Đã tải xong {files_count} files vào '{LOCAL_DESTINATION_FOLDER}'")
            return True
        elif not found_any:
            print(f"Không tìm thấy folder '{GCS_SOURCE_FOLDER}' trên bucket.")
            return False
        else:
            return True

    except Exception as e:
        print(f"Lỗi khi tải model: {e}")
        if os.path.exists(LOCAL_DESTINATION_FOLDER):
            shutil.rmtree(LOCAL_DESTINATION_FOLDER)
        return False

# --- LIFESPAN (QUẢN LÝ VÒNG ĐỜI SERVER) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("\n--- Đang khởi động Server ---")

    if download_from_gcs_folder():

        print("Đang load model cho Toxic Service...")
        toxic_service.load_model()

        print("Đang load model cho Recommend Service...")
        recommend_service.load_model()

        print("Server đã sẵn sàng phục vụ!\n")
    else:
        print("Server chạy nhưng KHÔNG tải được model.\n")

    yield

    print("\n--- Server đang tắt ---")

app = FastAPI(
    title="Movie AI System (Toxic Check + Recommender)",
    lifespan=lifespan
)

app.include_router(toxic_service.router)
app.include_router(recommend_service.router)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=5000, reload=True)