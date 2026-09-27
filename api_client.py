import requests

TIMEOUT_SECONDS = 30

class ApiClient:
    def __init__(self, base_url: str, function_key: str):
        self._base_url = base_url.rstrip("/")
        self._headers = {"x-functions-key": function_key}

    def upload_document(self, user_id: str, filename: str, file_bytes: bytes, content_type: str) -> dict:
        response = requests.post(
            f"{self._base_url}/api/documents",
            params={"user_id": user_id, "filename": filename},
            data=file_bytes,
            headers={**self._headers, "Content-Type": content_type},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def get_document_status(self, user_id: str, doc_id: str) -> dict:
        response = requests.get(
            f"{self._base_url}/api/documents/{doc_id}",
            params={"user_id": user_id},
            headers=self._headers,
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()