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

import toxic_service
import recommend_service

# --- CẤU HÌNH GCS ---
GCS_SOURCE_FOLDER = "model"
LOCAL_DESTINATION_FOLDER = "./model"
BUCKET_NAME = "comment-filter"

gcp_credentials_str = r"""
{
  "type": "service_account",
  "project_id": "lateral-vision-480110-e7",
  "private_key_id": "ca2a322046908177205ee83544c02594c9da1e16",
  "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQCuK6vT1M6P6z1V\nZsuHkYMOH2QU+kRGVxvH9SePDxSvOOolcghCgclYOWpW//HUcxE5yRFxfeWDeVIz\nUfxwt7omaxSAW//pM2lutnro9e0Q6Io5PP5wvQ8xQ+sHnXRLudrbEPex4amGjtEu\nznl9pAqNRPEjoga0t6M/oaRUeM4BumaOmx7BvqGrkedty3LWhkEP/+nDBj8WbzdP\nnfhQ8bKRw7auLJC9HouWIuYxYPmu3VSq/VkA8TW4ZcwfDFEpzLJv5PUYyHYdkc2W\nGnDsOrbLuV7j02Aph/yMR04wKAiHNKyES1iLW0Sh82VfNMNVJWgnNQPrlsVwjDw7\nupOYIdknAgMBAAECggEAICMNuvHmxVZuBD6J6nNWf7oaEObWxzXWefBJwtNRCT3S\nsmMWlBL3kFBTx0bFm+I5ePSZQ9wrh7VQjgigMDouXl1etIqFL0Xdu+Bu0GJkQjzu\nGD6hYjL4RSpXiRmr9jcOY2J/mbJeQeZDQcQ2VZD5o3RnnCAO5bSGqjyMokuCE34/\n1Yg5VrqK65Jvr83YbzNO26W2tTu7qgFEHABtqMrzMZcCOw/PeCzwGYGn7M2CAcID\nQpsKyYYCD7XHVbcf2Fmv/z4D5enqZY1yIeowIxVTHlUcPYBjGayu+LkSIh2zoF75\nkHSz7iERaLw1eHFdvoEw6VnccHn0ri/uycFfwpN6cQKBgQDfp8ZZYlJu5bf7Jl1S\ng+YyCCq8UtFohptDZ6mPj/6aRZYN33kzyBfpAdDFbwi0MhuUyZ03aYSxVhf3ScMF\nuFJ06KvuNFVKesqbBUgOk0pLoIwno4Gjq50GGcJKOa4Yy15U1DUBK1CgJNbnafJE\nscd8ZnN7DV6xYY5Gdo0M2+FedwKBgQDHW9vFPHO7VBX/lMOYfn7HK2LvlIvUFHMm\nXATt2IJ4UxNWj1ZngsBdk5MO6eGovOmczzykSLUAfUc2+mNKVRZv2eRVMRGPFVKR\nOGtI0jEkX1MOUP+oUL7MdVWWOS/bYz+noLZLdb2y5i/Tm/evnjacFruXG8CXOcux\nKhT39waW0QKBgED1vwOZLi4tpKlatEJSMxsXCiqWt2Hvrsr+Id6jySRPz8yJuu9S\nT7eKonOl01ZbM22cvYApsrO7OMzHbNCNf82bAsz2AEvYrF4oQ4yQNUUQBYVB1VaP\nctUhyC+83xcugmCLHjYPuaQ57v1Z5VcUd8dnDmWQNY+5sRRAPKmed6/zAoGBAIMm\nURweOtyf4qDy0wI9JVYNmtaV2K86jM5NdwipiJtzu6MURPlvsPFepj4HaubA8May\nJujE1B5wTCtE6ZD7DPmkVGwfDbgdhOX4Qcv0S1PsSg1/B9FI3VxQTG+5S1x+nF/A\ngGyAFr3cZNZHItirsq1Y3yv3m+lgojn4vzapkfIhAoGAcR4w8NQHAJeBWTaA4ZjM\nM8NlAwP6VfLFMGzWMa7UIuq1q35JODBDn4Krf8688+r+wpVKblUExpwLtYkgUX+N\n2AScU7qkPjBJIG+aLAxpNelDSFcioeyKOB46//EBY7TUbGvw0WoMHnSUd582tRSc\nzeFFJttBlSJu5aaHfUgWAMo=\n-----END PRIVATE KEY-----\n",
  "client_email": "railway-download@lateral-vision-480110-e7.iam.gserviceaccount.com",
  "client_id": "107331976707568869762",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/railway-download%40lateral-vision-480110-e7.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
"""

def download_from_gcs_folder():
    if os.path.exists(LOCAL_DESTINATION_FOLDER) and os.listdir(LOCAL_DESTINATION_FOLDER):
        print(f"Model đã có sẵn tại {LOCAL_DESTINATION_FOLDER}. Bỏ qua tải xuống.")
        return True

    print(f"Đang kết nối GCS để tải folder '{GCS_SOURCE_FOLDER}'...")

    if "BEGIN PRIVATE KEY" not in gcp_credentials_str:
        print("Lỗi: Bạn chưa dán Key vào biến gcp_credentials_str trong code!")
        return False

    try:
        # 2. Kết nối GCS
        creds_dict = json.loads(gcp_credentials_str)
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