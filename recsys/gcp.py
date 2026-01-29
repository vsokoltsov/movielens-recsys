from dataclasses import dataclass, field
import numpy as np
import json
import tempfile
from google.cloud import storage
from scipy.sparse import csr_matrix
from implicit.als import AlternatingLeastSquares


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


@dataclass
class GCPModelStorage:
    bucket_name: str
    client: storage.Client = field(init=False, repr=False)
    bucket: storage.Bucket = field(init=False, repr=False)

    def __post_init__(self):
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name=self.bucket_name)

    async def save_als_model(self, path: str, model: AlternatingLeastSquares) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            model.save(tmp.name)

            blob = self.bucket.blob(path)
            blob.upload_from_filename(tmp.name)

    async def load_als_model(self, path: str) -> AlternatingLeastSquares:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            blob = self.bucket.blob(path)
            blob.download_to_filename(tmp.name)
            model = AlternatingLeastSquares()
            return model.load(tmp.name)

    async def save_csr_npz(self, path: str, matrix: csr_matrix) -> None:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            np.savez_compressed(
                tmp.name,
                data=matrix.data,
                indices=matrix.indices,
                indptr=matrix.indptr,
                shape=matrix.shape,
            )
            blob = self.bucket.blob(path)
            blob.upload_from_filename(tmp.name)

    async def load_csr_npz(self, path: str) -> csr_matrix:
        with tempfile.NamedTemporaryFile(suffix=".npz") as tmp:
            blob = self.bucket.blob(path)
            blob.download_to_filename(tmp.name)
            loader = np.load(tmp.name, allow_pickle=False)
            return csr_matrix(
                (loader["data"], loader["indices"], loader["indptr"]),
                shape=tuple(loader["shape"]),
            )

    async def save_json(self, path: str, data: dict) -> None:
        blob = self.bucket.blob(path)
        blob.upload_from_string(json.dumps(data), content_type="application/json")

    async def load_json(self, path: str) -> dict:
        blob = self.bucket.blob(path)
        return json.loads(blob.download_as_text())

    async def save_onnx(self, path: str, local_onnx_path: str) -> None:
        blob = self.bucket.blob(path)
        blob.upload_from_filename(local_onnx_path)

    async def load_onnx(self, path: str) -> str:
        with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
            blob = self.bucket.blob(path)
            blob.download_to_filename(tmp.name)
            return tmp.name
