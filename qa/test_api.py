
"""
PathSnap API integration and QA tests.

Prerequisite:
    Start the API in another terminal:
    uvicorn api.main:app --host 127.0.0.1 --port 8000

Run:
    python -m unittest qa.test_api -v
"""

import json
import unittest
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


BASE_URL = "http://127.0.0.1:8000"

EXPECTED_TRIPS = {
    "00000000", "00000007", "00000009", "00000040",
    "00000052", "00000056", "00000065", "00000076",
    "00000077", "00000091",
}

FORBIDDEN_KEYS = {
    "label",
    "ref_arc",
    "reference_route",
    "ground_truth",
}


def get_response(path):
    """Send a GET request and return status, headers and JSON."""
    request = Request(f"{BASE_URL}{path}", method="GET")

    with urlopen(request, timeout=60) as response:
        body = response.read().decode("utf-8")
        return response.status, response.headers, json.loads(body)


def post_prediction(trip_id):
    """Send a prediction request and return its response."""
    payload = json.dumps({"trip_id": trip_id}).encode("utf-8")

    request = Request(
        f"{BASE_URL}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=180) as response:
        body = response.read().decode("utf-8")
        return response.status, response.headers, json.loads(body)


def find_forbidden_keys(value):
    """Find sensitive keys at any depth in a JSON response."""
    found = set()

    if isinstance(value, dict):
        for key, item in value.items():
            if key in FORBIDDEN_KEYS:
                found.add(key)
            found.update(find_forbidden_keys(item))

    elif isinstance(value, list):
        for item in value:
            found.update(find_forbidden_keys(item))

    return found


class TestPathSnapAPI(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Check that the API is reachable before running tests."""
        try:
            status, _, body = get_response("/health")
            if status != 200 or body.get("status") != "ok":
                raise RuntimeError("API health check failed")
        except (URLError, TimeoutError) as exc:
            raise RuntimeError(
                "API is not running. Start uvicorn first."
            ) from exc

    def test_01_health_endpoint(self):
        status, headers, body = get_response("/health")

        self.assertEqual(status, 200)
        self.assertEqual(body.get("status"), "ok")
        self.assertEqual(
            headers.get("X-PathSnap-Pipeline"), "trained"
        )

    def test_02_available_trips(self):
        status, headers, body = get_response("/trips")

        self.assertEqual(status, 200)
        self.assertEqual(
            headers.get("X-PathSnap-Pipeline"), "trained"
        )

        actual_trips = set(body.get("trips", []))

        self.assertEqual(actual_trips, EXPECTED_TRIPS)
        self.assertEqual(len(actual_trips), 10)

    def test_03_invalid_trip_returns_404(self):
        payload = json.dumps({
            "trip_id": "trip-does-not-exist"
        }).encode("utf-8")

        request = Request(
            f"{BASE_URL}/predict",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with self.assertRaises(HTTPError) as error:
            urlopen(request, timeout=60)

        self.assertEqual(error.exception.code, 404)

    def test_04_all_student_test_predictions(self):
        """Run and validate predictions for all 10 trips."""

        _, _, trips_body = get_response("/trips")
        trip_ids = trips_body["trips"]

        for trip_id in trip_ids:
            with self.subTest(trip_id=trip_id):
                status, headers, data = post_prediction(trip_id)

                self.assertEqual(status, 200)

                # Never accept placeholder results.
                self.assertEqual(
                    headers.get("X-PathSnap-Pipeline"),
                    "trained",
                )

                # Basic response structure.
                self.assertEqual(data.get("trip_id"), trip_id)

                self.assertIn("gps", data)
                self.assertIn("predicted", data)
                self.assertIn("route", data)
                self.assertIn("metrics", data)

                gps = data["gps"]
                predictions = data["predicted"]

                self.assertGreater(len(gps), 0)

                # Exactly one prediction per GPS observation.
                self.assertEqual(len(gps), len(predictions))

                # GPS coordinate validation.
                for point in gps:
                    self.assertEqual(len(point), 2)

                    lat, lon = point
                    self.assertTrue(-90 <= lat <= 90)
                    self.assertTrue(-180 <= lon <= 180)

                # Prediction structure and arc IDs.
                for prediction in predictions:
                    self.assertIn("obs_idx", prediction)
                    self.assertIn("arc_id", prediction)

                    self.assertIsInstance(
                        prediction["obs_idx"], int
                    )
                    self.assertIsInstance(
                        prediction["arc_id"], int
                    )

                    self.assertGreaterEqual(
                        prediction["arc_id"], -1
                    )

                # Check evaluation metrics.
                metrics = data["metrics"]

                required_metrics = {
                    "top1_accuracy",
                    "route_continuity",
                    "route_f1",
                    "candidate_miss_rate",
                    "inference_ms",
                }

                self.assertTrue(
                    required_metrics.issubset(metrics.keys())
                )

                for name in required_metrics - {"inference_ms"}:
                    value = metrics[name]

                    if value is not None:
                        self.assertTrue(0 <= value <= 1)

                self.assertIsNotNone(metrics["inference_ms"])
                self.assertGreaterEqual(
                    metrics["inference_ms"], 0
                )

                # Reference and ground-truth privacy.
                exposed = find_forbidden_keys(data)

                self.assertEqual(
                    exposed,
                    set(),
                    f"Sensitive fields exposed: {exposed}",
                )

                print(
                    f"\nPASS {trip_id}: "
                    f"{len(gps)} GPS points, "
                    f"{len(predictions)} predictions, "
                    f"Top-1={metrics['top1_accuracy']}"
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
