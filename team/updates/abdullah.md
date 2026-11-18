# Abdullah's updates

Only Abdullah edits this file. Append new entries at the bottom.
## 2026-11-18

### Done
- Completed Task 1: `eda/inventory.py`.
- Verified 80 training trips and 10 student_test trips.
- Confirmed all expected files are present and there are no missing values.
- Generated the trip inventory CSV and GPS points histogram under `PATHSNAP_WORK/eda`.

### Next
- Task 2: GPS quality analysis.

### Blocked
- None.

## 2026-11-18 — Task 2: GPS Quality Analysis

### Done
- Created `eda/gps_quality.py`.
- Analysed all 90 trips (80 train and 10 student_test).
- Implemented NumPy Haversine distance calculation in metres.
- Calculated GPS time gaps, distances, speeds and total trip distances.
- Checked one-second sampling intervals and non-increasing timestamps.
- Recorded invalid rows removed and extreme speed intervals above 200 km/h.
- Confirmed zero invalid rows and zero non-increasing timestamps.
- Verified the largest time gap of 153064 seconds in trip 00000090.
- Found 921 intervals above the exploratory 200 km/h threshold.
- Generated GPS time-gap and speed distribution plots.
- Saved outputs under PATHSNAP_WORK/eda.

### Findings for Shriram
- Very high GPS speeds were observed in trips 00000001, 00000013 and 00000032.
- Trip 00000032 has 610 intervals above 200 km/h.
- Trips 00000090 and 00000078 have very large time gaps with minimal coordinate displacement.
- High-speed observations were retained, not removed.

### Next
- Task 3: Road Network Analysis.

### Blocked
- None.