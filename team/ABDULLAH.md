# Abdullah's brief: data analysis, API testing and report figures

You own `eda/`, `qa/` and `report/`. This is real coding work that counts toward the rubric, and nobody is blocking you, so Tasks 1 to 5 can start now. You will not have Claude, so each task says what to build, what it must produce, and how to check it is right.

You will be asked to explain your code in the viva. Write it so you can defend every line.

## Setup

```
git clone https://github.com/Shriram60/ML-Path_Snap-Map-Matching.git
cd ML-Path_Snap-Map-Matching
python -m venv .venv
.venv\Scripts\activate
pip install numpy pandas scipy matplotlib networkx folium requests
set PATHSNAP_DATA=<folder containing train and student_test>
```

Read the data path from `os.environ["PATHSNAP_DATA"]`. Never hardcode your own path, and never commit the dataset.

Git routine: `git pull --rebase`, commit, `git push`. Edit only your folders. Commit messages are short with no punctuation, for example `add trip inventory`.

Post progress in `team/updates/abdullah.md`. The others' Claudes read it.

```
## HH:MM
Done: what works and how to run it
Next: what you are doing now
Problem: what broke, or none
To Shriram: a request, only if needed
```

## The data

| File | Columns, no header, separated by whitespace |
|---|---|
| `<id>.track` | longitude, latitude, timestamp in seconds |
| `<id>.nodes` | longitude, latitude. Row number is the node ID |
| `<id>.arcs` | from_node, to_node. Row number is the arc ID |
| `<id>.route` | arc ID, in the order driven |

Save outputs in `eda/outputs/`, and put the partition (`train` or `student_test`) in every table and plot title.

## Task 1: Dataset inventory, `eda/inventory.py`

For every trip, write one row to `trip_inventory.csv`: partition, trip ID, which files exist, file sizes, GPS points, nodes, arcs, route length, missing values, min and max longitude and latitude.

Also save a histogram of GPS points per trip.

Check: 80 train trips, 10 student-test trips, all 4 files in each, no missing values, and GPS points per trip from 262 to 19313.

## Task 2: GPS quality, `eda/gps_quality.py`

Write your own `haversine(lon1, lat1, lon2, lat2)` using NumPy arrays, returning metres. Do not use a library for it, and be ready to explain why plain distance in degrees is wrong.

For each trip compute time gaps, step distance, speed, share of 1-second steps, largest gap, non-increasing timestamps, and total GPS distance. Save `gps_quality.csv`, plus plots of the time-gap and speed distributions.

Find trips with large time gaps or impossible speeds and list them. Look at whether a large time gap also means a large jump in position.

Check: timestamps never go backwards, and the largest gap is 153064 seconds, in trip 00000090.

## Task 3: Road network, `eda/road_network.py`

Use `networkx`. For each trip build a directed graph and report: nodes, arcs, weakly connected components, share of nodes in the largest component, isolated nodes, self-loops, duplicate arcs, share of arcs with a reverse arc, and average in and out degree.

Then check each route: arc `i` is connected to arc `i+1` when the `to_node` of `i` equals the `from_node` of `i+1`. Report the share of connected steps, and the arcs repeated in the route.

Save `road_network.csv` and at least one plot.

Check: no isolated nodes, and 11 of the 90 routes have at least one disconnected step.

## Task 4: GPS against the route, `eda/route_fit.py` (the hardest one)

For each GPS point, compute the distance in metres to the nearest segment of that trip's route.

1. Convert longitude and latitude to local metres around the trip's centre: `x = R * radians(lon - lon0) * cos(radians(lat0))`, `y = R * radians(lat - lat0)`, with `R = 6371000`.
2. For point `P` and segment `A` to `B`: `u = clip(dot(P-A, B-A) / dot(B-A, B-A), 0, 1)`, nearest point `C = A + u(B-A)`, distance `|P-C|`. Guard against zero-length segments.
3. Vectorise with NumPy where you can, because some trips are large.

Save median, 95th percentile and maximum distance per trip in `route_fit.csv`, the share of points over 50 m, and the first and last point over 50 m. Plot the distance distribution.

Check: the typical trip has a median of about 4.5 m. Trips 00000054 and 00000033 have medians around 138 m and 122 m, because their GPS carries on after the reference route ends. Trip 00000002 has a similar tail. Tell Shriram which other trips behave this way, as this directly affects labelling.

## Task 5: Folium maps, `eda/trip_map.py`

Running `python eda/trip_map.py --trip 00000033` should save an HTML map with separate layers for GPS points, nearby road network, and reference route, plus start and end markers, a legend and `LayerControl`.

Folium needs `[lat, lon]`, the opposite of the files. Draw only roads near the track and thin out GPS points on long trips, or the map will be too heavy to open.

Make maps for at least 5 trips, including 00000002, 00000033 and 00000054. These are for train and student-test trips only.

## Task 6: API test script, `qa/test_api.py`

Build this against the API contract in `CLAUDE.md`, then run it once Shriya's API is up:

```
python qa/test_api.py --base-url http://<laptop-1-ip>:8000
```

It must:

- Call `/health`, `/trips`, then `/predict` for every trip
- Check that all required fields exist with the right types
- Check that `predicted` arc IDs are integers and `obs_idx` values are not repeated
- Check that coordinates are valid and in `[lat, lon]` order
- Check that metrics are between 0 and 1 where they should be
- Fail if the response contains anything like `label`, `ref_arc`, `reference` or `ground_truth`
- Send an invalid trip ID and confirm the API returns a clean error
- Record the response time and result per trip in `qa/api_test_results.csv`
- Exit with an error code if any check fails

Test it first against a tiny fake FastAPI server of your own, including one bad response, to prove it can catch failures. Keep fake responses out of any reported results.

## Task 7: Report figures, `report/make_figures.py`

When Sharanya and Shriram have committed result CSVs, generate the required charts directly from those files:

- Cross-validation Macro F1 across models
- Student-test Top-1 accuracy and Macro F1
- Log loss, where it applies
- Untuned against tuned model
- Classifier only against classifier plus HMM

Agree the CSV column names with them through your update file. Never type result values into the script.

## Task 8: Prove it works

- Run all 10 student-test trips through the real UI and confirm each request appears in the API window on Laptop 1.
- Follow the README from a clean folder and report every step that fails.
- Take real screenshots of the UI, the API logs and your maps for the report.

## Report

Write the EDA part of section 2.2 from your own outputs: missing values, GPS plots, road network and Folium maps. Then assemble the team's sections into one PDF in the order given in `Deliverables.pdf`.

## Stretch goal

Once Shriram's candidate table is ready, and if Sharanya agrees, train one extra model such as distance-weighted KNN using Sharanya's training code and add it to the comparison. It must be split by trip, and you must be able to explain its results.

## Viva

Be ready to explain the file relationships, haversine, your point-to-segment distance, graph components, what the off-route tails mean for labels, and what your API test proves. Also learn the whole pipeline in outline: candidates, model scores, Viterbi, API, UI.
