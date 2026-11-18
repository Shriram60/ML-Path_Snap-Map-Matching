# 2.2 Dataset Understanding and Exploratory Data Analysis

## 2.2.1 Dataset Overview

The PathSnap dataset contains GPS trajectories and corresponding road-network information. Each trip consists of four files:

- `.track`: GPS longitude, latitude and timestamp.
- `.nodes`: Road-network node coordinates.
- `.arcs`: Directed connections between road-network nodes.
- `.route`: Reference route represented as an ordered sequence of arc IDs.

The dataset contains 90 trips, divided into 80 training trips and 10 student-test trips. All expected files were available for every trip.

The number of GPS observations varies considerably, ranging from 262 to 19,313 points per trip. This variation indicates that the dataset contains trajectories of different lengths and recording durations.

## 2.2.2 Missing-Value and GPS Data-Quality Analysis

An inventory analysis was performed to verify the availability and completeness of the dataset files. No missing values were detected in the inspected trip data.

GPS quality was examined using timestamps, consecutive-point distances, time gaps and calculated speeds. The Haversine distance formula was implemented using NumPy to calculate geographical distances in metres between consecutive GPS observations.

The analysis produced the following observations:

- No invalid GPS rows were detected by the EDA quality check.
- No non-increasing timestamps were detected.
- The median time gap was one second.
- The median share of one-second sampling intervals was approximately 99.14%.
- The largest observed time gap was 153,064 seconds in trip 00000090.
- A total of 921 GPS intervals exceeded the exploratory speed threshold of 200 km/h.

High-speed observations were retained rather than automatically removed. Such observations can arise from GPS noise, coordinate jitter or unusual recording behaviour and therefore require investigation before being classified as invalid.

Trip 00000077 was identified as an example where GPS jitter while stationary can produce misleading apparent speeds.

The quality analysis helped identify unusual observations and understand the limitations of the raw GPS trajectories before map matching.

## 2.2.3 Road-Network Analysis

The road network was analysed using NetworkX. Each trip's road network was represented as a directed graph, where nodes represent geographical points and arcs represent directed road connections.

The analysis examined graph connectivity, weakly connected components, largest-component share, isolated nodes, self-loops, duplicate arcs and reference-route connectivity.

The main findings were:

- No isolated nodes were detected.
- The average share of nodes belonging to the largest weakly connected component was approximately 96.84%.
- Eleven trips contained at least one disconnected transition in their reference route.
- Fourteen disconnected route transitions were identified across these trips.
- The average connected-step share across reference routes was approximately 99.94%.
- The analysis identified 1,903 duplicate arc rows and 24 self-loops across the dataset.

These results indicate that most consecutive reference-route arcs are connected, although a small number of trips contain discontinuities that must be considered during route reconstruction.

Duplicate arc rows were counted separately from graph connectivity calculations because a NetworkX directed graph represents repeated connections between the same node pair as a single graph edge.

**Figure to include:** Road-network analysis chart.

## 2.2.4 GPS Trajectory Fit Against Reference Routes

The GPS observations were compared against their corresponding reference-route geometries to understand how closely the recorded trajectories followed the provided routes.

Longitude and latitude coordinates were converted into local metre-based coordinates around each trip's geographical centre. A vectorised point-to-segment distance calculation was then implemented using NumPy. Zero-length route segments were handled separately to avoid division by zero.

The distance to the nearest reference-route segment was calculated for every GPS observation. The analysis recorded median, 95th-percentile and maximum distances, as well as the proportion of points more than 50 metres away from the reference route.

The results showed:

- The median of the trip-level median distances was approximately 4.52 metres.
- The mean of the trip-level median distances was approximately 7.74 metres.
- The average trip-level 95th-percentile distance was approximately 22.89 metres.
- Twenty-three trips contained at least one GPS point more than 50 metres from the reference route.
- A total of 22,581 GPS observations were more than 50 metres from their reference route.
- The maximum observed distance was approximately 569.49 metres.
- Two trips had median GPS-to-reference distances greater than 50 metres.

Some trips showed particularly unusual patterns:

| Trip ID | Median distance | Points beyond 50 m |
|---|---:|---:|
| 00000054 | 138.51 m | 87.19% |
| 00000033 | 122.38 m | 58.80% |
| 00000002 | 5.66 m | 11.98% |

Trips 00000033 and 00000054 showed substantial deviations from their reference routes. Trip 00000002 had a relatively low median distance but a large upper-distance tail, with a maximum observed distance of approximately 569.49 metres.

The analysis also identified four trips whose first GPS observation and four trips whose last GPS observation were more than 50 metres from their reference route.

These observations are important because GPS points recorded outside the supplied reference route may not have reliable corresponding reference labels. Such cases must be considered when preparing training labels and evaluating route-matching performance.

**Figure to include:** GPS-to-reference route-fit analysis.

## 2.2.5 Geographical Exploration Using Folium

Interactive geographical maps were generated using Folium to visually compare GPS trajectories with reference routes and nearby road-network geometry.

Five representative trips were selected:

- 00000002
- 00000033
- 00000054
- 00000043
- 00000096

The maps include separate layers for GPS trajectories, reference routes and nearby road-network segments, together with start and end markers, a legend and layer controls.

Nearby road segments were filtered geographically around each trip. A maximum of 15,000 road segments was displayed to control map complexity.

The maps were generated as interactive HTML files and saved outside the repository under:

`$PATHSNAP_WORK/eda/maps/`

These visualisations helped investigate the relationship between observed GPS trajectories and the supplied road network, particularly for trips with unusual route-fit measurements.

The current EDA maps show GPS observations, reference routes and nearby roads. Predicted-route visualisations generated through the final map-matching pipeline are covered separately in Section 6.

## 2.2.6 Summary of EDA Observations

The exploratory analysis established that the dataset contains complete trip files, generally consistent GPS sampling and mostly connected road networks. However, several trips contain large time gaps, extreme apparent speeds or substantial deviations from their reference routes.

The route-fit analysis was particularly useful in identifying trips where GPS observations deviated significantly from the supplied reference route. These findings provide important context for data preparation, reference-label reliability and subsequent map-matching evaluation.

The EDA findings were shared with the data-processing and modelling workstreams to help the team understand unusual GPS behaviour and potential challenges in route reconstruction.