"""Create portable Postman tests with placeholders, never embedded API credentials."""
import json
from pathlib import Path


def request_item(name, method, path, file=None, status=200, extra_tests=None):
    request = {"method": method, "header": [{"key": "Authorization", "value": "Bearer {{api_key}}"}],
               "url": {"raw": "{{base_url}}" + path, "host": ["{{base_url}}"],
                       "path": path.strip("/").split("/")}}
    if file:
        request["body"] = {"mode": "formdata", "formdata": [
            {"key": "image", "type": "file", "src": "docs/samples/" + file}]}
    lines = [f"pm.test('HTTP {status}', function () {{ pm.response.to.have.status({status}); }});"]
    if extra_tests:
        lines.extend(extra_tests)
    return {"name": name, "request": request,
            "event": [{"listen": "test", "script": {"type": "text/javascript", "exec": lines}}]}


if __name__ == "__main__":
    prediction_tests = [
        "const body = pm.response.json();",
        "pm.test('Valid prediction contract', function () {",
        "  pm.expect(['real', 'ai_generated']).to.include(body.label);",
        "  pm.expect(body.confidence).to.be.within(0, 1);",
        "  pm.expect(body.model_version).to.be.a('string');",
        "  pm.expect(body.image_sha256).to.have.lengthOf(64);",
        "  pm.expect(body.id).to.be.a('string');",
        "});",
        "pm.collectionVariables.set('last_prediction_id', body.id);",
    ]
    items = [request_item("1. Health", "GET", "/health"),
             request_item("2. Real example", "POST", "/api/predict", "real.jpg",
                          extra_tests=prediction_tests),
             request_item("3. AI-generated example", "POST", "/api/predict", "ai_generated.jpg",
                          extra_tests=prediction_tests),
             request_item("4. Database history", "GET", "/api/history", extra_tests=[
                 "const rows = pm.response.json().predictions;",
                 "pm.test('Prediction was persisted', function () {",
                 " pm.expect(rows.map(x => x.id)).to.include(pm.collectionVariables.get('last_prediction_id'));",
                 "});"]),
             request_item("5. Reject unsupported type", "POST", "/api/predict", "wrong_type.txt", 415),
             request_item("6. Reject undecodable image", "POST", "/api/predict", "undecodable.png", 400),
             request_item("7. Reject oversized upload", "POST", "/api/predict", "oversized.png", 413)]
    collection = {"info": {"name": "22i-1714_A2_Ops VisionGuard API",
                            "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"},
                  "variable": [{"key": "base_url", "value": "http://127.0.0.1:8000"},
                               {"key": "api_key", "value": ""}], "item": items}
    target = Path("docs/postman")
    target.mkdir(parents=True, exist_ok=True)
    (target / "VisionGuard.postman_collection.json").write_text(json.dumps(collection, indent=2))
    (target / "README.md").write_text(
        "# Postman verification\n\nImport the collection, set base_url to the hosted API base URL "
        "and api_key to your private API key, then select each file from docs/samples. "
        "Run the requests in order. The history assertion checks that the last prediction was saved. "
        "The last three requests intentionally expect 415, 400 and 413.\n\n"
        "Newman can execute the same collection with the project root as its working directory. "
        "Keep any environment containing the real API key out of Git and screenshots.\n")
    print("Postman collection created with credential placeholders.")
