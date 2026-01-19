from dataclasses import dataclass, field
from google.cloud import storage

@dataclass
class GCPStorageClient:
    client: storage.Client = field(default_factory=storage.Client)

    def __post_init__(self):
        self.client = storage.Client()

    def download(self, bucket_name: str, object_name: str, dst_path: str) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        blob.download_to_filename(dst_path)
        print(f"Downloaded gs://{bucket_name}/{object_name} -> {dst_path}")

    def upload(self, path: str, bucket_name: str, object_name: str) -> None:
        bucket = self.client.bucket(bucket_name)
        blob = bucket.blob(object_name)

        blob.upload_from_filename(path)
        print(f"Uploaded {path} -> gs://{bucket_name}/{object_name}")

    def read_bytes(self, bucket: str, obj: str) -> bytes:
        blob = self.client.bucket(bucket).blob(obj)
        return blob.download_as_bytes()