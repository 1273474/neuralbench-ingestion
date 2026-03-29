"""
Integration test for the TRIBE v2 inference pipeline.
Requires MOCK_INFERENCE=true (default) for local execution.
"""
import os
os.environ["MOCK_INFERENCE"] = "true"  # Ensure mock mode for testing

from fastapi.testclient import TestClient
from main import app
import json

client = TestClient(app)

# ── Test 1: Health Check ───────────────────────────────────────
print("=" * 60)
print("TEST 1: Health Check")
print("=" * 60)
r = client.get("/health")
print(f"  Status: {r.status_code}")
print(f"  Body:   {json.dumps(r.json(), indent=2)}")
assert r.status_code == 200
print("  ✅ PASSED\n")

# ── Test 2: Analyze with default demographic (Gen-Z) ──────────
print("=" * 60)
print("TEST 2: POST /analyze-media (demographic=Gen-Z)")
print("=" * 60)
with open("sample_test.mp4", "rb") as f:
    r = client.post(
        "/analyze-media",
        files={"file": ("sample_test.mp4", f, "video/mp4")},
        data={"demographic": "Gen-Z"},
    )
print(f"  Status: {r.status_code}")
print(f"  Body:   {json.dumps(r.json(), indent=2)}")
assert r.status_code == 200
metrics = r.json()
assert 0 <= metrics["visual_intensity"] <= 100
assert 0 <= metrics["audio_complexity"] <= 100
assert 0 <= metrics["text_density"] <= 100
print("  ✅ PASSED\n")

# ── Test 3: Analyze with Academic demographic ──────────────────
print("=" * 60)
print("TEST 3: POST /analyze-media (demographic=Academic)")
print("=" * 60)
with open("sample_test.mp4", "rb") as f:
    r = client.post(
        "/analyze-media",
        files={"file": ("sample_test.mp4", f, "video/mp4")},
        data={"demographic": "Academic"},
    )
print(f"  Status: {r.status_code}")
print(f"  Body:   {json.dumps(r.json(), indent=2)}")
assert r.status_code == 200
print("  ✅ PASSED\n")

# ── Test 4: Analyze with Professional demographic ──────────────
print("=" * 60)
print("TEST 4: POST /analyze-media (demographic=Professional)")
print("=" * 60)
with open("sample_test.mp4", "rb") as f:
    r = client.post(
        "/analyze-media",
        files={"file": ("sample_test.mp4", f, "video/mp4")},
        data={"demographic": "Professional"},
    )
print(f"  Status: {r.status_code}")
print(f"  Body:   {json.dumps(r.json(), indent=2)}")
assert r.status_code == 200
print("  ✅ PASSED\n")

# ── Test 5: Invalid demographic should return 400 ─────────────
print("=" * 60)
print("TEST 5: Invalid demographic → 400")
print("=" * 60)
with open("sample_test.mp4", "rb") as f:
    r = client.post(
        "/analyze-media",
        files={"file": ("sample_test.mp4", f, "video/mp4")},
        data={"demographic": "Teenager"},
    )
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.json()}")
assert r.status_code == 400
print("  ✅ PASSED\n")

# ── Test 6: Invalid file extension should return 400 ──────────
print("=" * 60)
print("TEST 6: Invalid extension → 400")
print("=" * 60)
r = client.post(
    "/analyze-media",
    files={"file": ("document.pdf", b"fake content", "application/pdf")},
    data={"demographic": "Gen-Z"},
)
print(f"  Status: {r.status_code}")
print(f"  Body:   {r.json()}")
assert r.status_code == 400
print("  ✅ PASSED\n")

print("=" * 60)
print("ALL 6 TESTS PASSED ✅")
print("=" * 60)
