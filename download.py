import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./steam-form-422902-h7-6c70a0e3a345.json"

from google.cloud import storage

bucket_name = 'recipable'    # 서비스 계정 생성한 bucket 이름 입력
source_blob_name = 'model.pth'    # GCP에 저장되어 있는 파일 명
destination_file_name = './model.pth'    # 다운받을 파일을 저장할 경로("local/path/to/file")

storage_client = storage.Client()
bucket = storage_client.bucket(bucket_name)
blob = bucket.blob(source_blob_name)

blob.download_to_filename(destination_file_name)