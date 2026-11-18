# Running PathSnap on two laptops

| Laptop | Runs | Needs |
|---|---|---|
| Laptop 1 | API, model, HMM, road network, server-side metrics | Python, this repo, the data folder |
| Laptop 2 | UI in a browser | Python (only to serve the `ui/` folder), this repo |

The UI talks to the API over the local network. It receives only the trip ID list and the `/predict` response. Reference routes and labels stay on Laptop 1.

```
Laptop 2 browser ──GET /trips──────────▶ Laptop 1 API ──▶ list folders in PATHSNAP_EVAL_DIR
                 ──POST /predict {id}──▶ load_trip → hide route → build_candidates
                                         → score (trained model) → decode (Viterbi)
                                         → metrics against the hidden route
                 ◀── gps, predicted arcs, route polyline, metrics ──
```

## 1. Both laptops: get the code

```
git clone https://github.com/Shriram60/ML-Path_Snap-Map-Matching.git
cd ML-Path_Snap-Map-Matching
```

## 2. Laptop 1: install and start the API

```
python -m venv .venv
.venv\Scripts\activate
pip install -r deployment\requirements-api.txt
set PATHSNAP_DATA=C:\path\to\folder\with\train_and_student_test
deployment\run_api.bat
```

On demo day, point the API at the evaluator trips instead of `student_test`:

```
set PATHSNAP_EVAL_DIR=C:\path\to\evaluator\trips
```

The console should show the evaluation folder, the number of trips, where each pipeline function comes from, and the model and HMM config files with their sizes. Keep this window visible. Evaluators check it, and every request is logged there with the client IP, status and time.

If startup says `Pipeline functions missing`, someone's code is not pulled yet. Run `git pull`.

## 3. Laptop 1: allow port 8000 through the firewall

Open CMD as administrator once:

```
netsh advfirewall firewall add rule name="PathSnap API 8000" dir=in action=allow protocol=TCP localport=8000
```

Or allow Python on **Private networks** when Windows asks the first time the API starts. Set the Wi-Fi network profile to **Private**.

Find Laptop 1's address with `ipconfig` (IPv4 Address, for example `192.168.1.20`).

## 4. Laptop 2: check the connection, then start the UI

In a browser on Laptop 2, open `http://192.168.1.20:8000/health`. It must show `{"status":"ok"}`. If it does not, the UI will not work either. See troubleshooting below.

```
deployment\run_ui.bat 192.168.1.20
```

Open `http://localhost:5500/index.html?api=http://192.168.1.20:8000`. The API address can also be typed into the box at the top of the UI, and it is remembered. The default lives in `ui/config.js`.

The map background needs internet for OpenStreetMap tiles. Without it, the GPS points and route still draw on a blank background.

## 5. Demo checklist

- [ ] `git pull` on both laptops. Both on the same commit.
- [ ] `api/placeholder.py` deleted, and the API console shows no `PLACEHOLDER` line.
- [ ] `models/artifacts/final_model.joblib` and `hmm/hmm_config.json` show sizes in the startup log, not `MISSING`.
- [ ] `PATHSNAP_EVAL_DIR` points at the evaluator trips. The UI lists exactly 10 trips.
- [ ] `/health` opens from Laptop 2's browser.
- [ ] Each of the 10 trips runs from the UI, and each request appears in the Laptop 1 console.

## Troubleshooting

| Symptom in the UI | Likely cause | Fix |
|---|---|---|
| Cannot reach the API | Different networks, firewall, wrong IP, API not running | Same Wi-Fi or hotspot, firewall rule above, re-check `ipconfig` |
| Works on Laptop 1 but not Laptop 2 | Firewall, or the network blocks device-to-device traffic (common on college Wi-Fi) | Use a phone hotspot both laptops join |
| No reply within 60 s | Pipeline slow or stuck on a trip | Watch the API console. Raise `PATHSNAP_TIMEOUT_S` in `ui/config.js` if it is just slow |
| API error 404 Unknown trip_id | Trip folder not in the evaluation folder | Check `PATHSNAP_EVAL_DIR` |
| API error 500 | A pipeline step raised an error | The traceback is in the API console. Send it to the owner of that step |
| Red placeholder banner | API is using `api/placeholder.py` | Delete it and pull the real modules |

## Run without a second laptop

Both can run on one machine for development: start the API, then `deployment\run_ui.bat` with no IP.
