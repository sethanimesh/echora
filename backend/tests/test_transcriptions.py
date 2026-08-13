from fastapi.testclient import TestClient

from app.main import app


def test_upload_retrieve_and_delete_transcription() -> None:
    with TestClient(app) as client:
        upload = client.post(
            "/v1/transcriptions",
            files={"file": ("example.wav", b"not-real-audio", "audio/wav")},
        )

        assert upload.status_code == 201
        created = upload.json()
        assert created["status"] == "completed"
        assert created["provider"] == "fake"
        assert created["language"] == "en"
        assert created["size_bytes"] == len(b"not-real-audio")
        assert created["hypotheses"] == [
            {
                "rank": 1,
                "text": "This is a deterministic fake transcript.",
                "score": 1.0,
                "score_type": "deterministic_fixture",
                "segments": [],
            }
        ]

        fetched = client.get(f"/v1/transcriptions/{created['id']}")
        assert fetched.status_code == 200
        assert fetched.json() == created

        deleted = client.delete(f"/v1/transcriptions/{created['id']}")
        assert deleted.status_code == 204
        assert client.get(f"/v1/transcriptions/{created['id']}").status_code == 404


def test_upload_rejects_unsupported_file_type() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/v1/transcriptions",
            files={"file": ("notes.txt", b"not audio", "text/plain")},
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported audio file extension."
