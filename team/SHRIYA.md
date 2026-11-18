# Shriya's brief: API, UI, maps and deployment

Read `CLAUDE.md` first. You make the system work across two laptops, which is what the evaluator sees.

## Start now, before the model is ready

Build the API and UI against the API contract using clearly marked placeholder pipeline code. Remove every placeholder before the demo, because hardcoded outputs are prohibited and the repo will be checked for them.

## Build, in this order

1. `api/`: FastAPI with `/health`, `/trips` and `/predict`, loading the artifacts once at startup and logging each request clearly in the console.
2. `api/`: the real pipeline: `load_trip`, `build_candidates`, `score`, `decode`, then server-side metrics.
3. `ui/`: list the trips from `/trips`, send the chosen trip ID, then show the route on a map with its metrics. The API address must be configurable.
4. `visualisations/`: Folium maps with GPS points, nearby roads, predicted route and, for development trips only, the reference route. Use distinct colours, a legend, sensible zoom and layer controls.
5. `deployment/`: requirements, run steps for both laptops, firewall and same-network notes. Help keep the README accurate.
6. Test all 10 student-test trips across two real laptops, including API failure and slow responses in the UI.

## Important limits

- The UI gets only the trip ID list and the `/predict` response.
- Reference routes and labels never go to the UI during evaluation.
- Do not hardcode the trip list. Read it from the configured data folder.

## Tell the team when

- The API and UI run end to end with placeholders: tell Shriram and Sharanya.
- A pipeline function is slow or fails on a trip: tell its owner, with the trip ID and error.

## Report sections

6 Folium visualisation, and 7 API, UI and deployment, with real screenshots of the UI and of API responses in CMD on the receiving laptop.

## Viva

Be ready to explain the request flow, where ground truth stays, how artifacts load, and how the two laptops connect.
