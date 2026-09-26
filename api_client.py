"""
Thin wrapper around the Function App's HTTP endpoints
(function_app.py: register_user, login, list_documents, upload_document,
get_document_status). Every method raises requests.HTTPError on a non-2xx
response via raise_for_status() -- callers in app.py catch that and show
an error rather than letting a raw traceback surface in the UI.

Auth: the function key goes in the x-functions-key HEADER, not the ?code=
query string -- both work (Azure Functions supports either), but the
header keeps the key out of URLs that might end up in request logs.
"""

import requests

_TIMEOUT_SECONDS = 30

class ApiClient:
    def __init__(self, base_url: str, function_key: str):
        self._base_url = base_url.rstrip("/")
        self._headers = {"x-functions-key": function_key}

    def upload_document(self, user_id: str, filename: str, file_bytes: bytes, content_type: str) -> dict:
        resp = requests.post(
            f"{self._base_url}/api/documents",
            params={"user_id": user_id, "filename": filename},
            data=file_bytes,
            headers={**self._headers, "Content-Type": content_type},
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()

    def get_document_status(self, user_id: str, doc_id: str) -> dict:
        resp = requests.get(
            f"{self._base_url}/api/documents/{doc_id}",
            params={"user_id": user_id},
            headers=self._headers,
            timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
        return resp.json()