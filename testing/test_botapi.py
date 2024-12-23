import pytest
from mediapi import app
from fastapi.testclient import TestClient

class Test_Emart:
    client = TestClient(app)

    @pytest.mark.skip
    def test_hello(self):
        response = self.client.get("/api/medibot/test")
        print("response: ", response.text)
        print("responsejson: ", response.json())
        assert response.status_code == 200
        assert response.json() == {"test": "hello medibot api"}