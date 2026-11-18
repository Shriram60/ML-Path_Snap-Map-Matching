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


## 2026-11-18 — Task 3: Road Network Analysis

### Done
- Created `eda/road_network.py`.
- Analysed all 90 trips: 80 train and 10 student_test.
- Built directed NetworkX graphs from the nodes and arcs files.
- Calculated node and arc counts, weakly connected components, largest component share, isolated nodes, self-loops, duplicate arcs, reverse arc share and average degrees.
- Calculated reference route connectivity and repeated route arcs.
- Confirmed zero isolated nodes.
- Found 11 trips with disconnected route steps, matching the expected team finding.
- Identified 14 disconnected route transitions across those 11 trips.
- Calculated an average largest component share of 0.9684.
- Generated road network CSV and connectivity plots.
- Saved outputs under PATHSNAP_WORK/eda.

### Findings for Shriram
- Disconnected route steps were found in trips 00000002, 00000019, 00000033, 00000041, 00000042, 00000069, 00000075, 00000080, 00000086, 00000089 and student_test trip 00000076.
- Found 1,903 duplicate arc rows and 24 self-loops across the dataset.
- Average route connected-step share was 0.9994.
- No isolated nodes were found.

### Next
- Task 4: Route Fit Analysis.

### Blocked
- None.
