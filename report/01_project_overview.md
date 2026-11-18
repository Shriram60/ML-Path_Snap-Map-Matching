# 1. Project Overview

## 1.1 Introduction

PathSnap is a machine learning-based GPS map-matching and route reconstruction system. Its purpose is to identify the road segments corresponding to noisy GPS observations and reconstruct the route travelled by a vehicle.

GPS devices frequently produce inaccurate location observations due to measurement noise, signal interference and other positioning errors. As a result, recorded GPS coordinates may not lie directly on the road network. Simply connecting consecutive GPS points may therefore produce an inaccurate representation of the actual route.

PathSnap addresses this problem through a combination of geographic data processing, feature engineering, machine learning and probabilistic route reconstruction.

The system first identifies nearby road segments for each GPS observation. A trained Random Forest model scores these candidate segments, and a Hidden Markov Model (HMM) with Viterbi decoding selects a likely sequence of connected road segments to reconstruct the route.

## 1.2 Problem Statement

Given a sequence of noisy GPS observations and a digital road network, the objective is to determine which road segments were most likely travelled by the vehicle.

The main challenges include:

- GPS coordinates may deviate from their actual road positions.
- Multiple nearby roads may be plausible candidates for the same GPS observation.
- GPS observations may contain irregular time intervals and sudden positional changes.
- Selecting the closest road independently for each observation can produce disconnected or unrealistic routes.
- The reconstructed route should maintain geographical and directional consistency.

Therefore, the problem requires both accurate road-segment classification and sequential route reconstruction.

## 1.3 Project Objectives

The main objectives of PathSnap are:

1. Analyse and understand the GPS trip data and road network.
2. Process GPS observations and prepare geographical information for map matching.
3. Generate candidate road segments for each GPS observation.
4. Extract meaningful geographical, directional, temporal and contextual features.
5. Train and compare machine learning models for candidate road-segment classification.
6. Apply Bayesian hyperparameter optimisation to improve model and candidate-generation performance.
7. Use an HMM and the Viterbi algorithm to reconstruct a consistent sequence of road segments.
8. Develop geographical visualisations to compare GPS observations and predicted routes.
9. Build an API and user interface for running the map-matching pipeline.
10. Test the complete system using the student-test trips and a two-laptop deployment setup.

## 1.4 Proposed Solution

PathSnap implements a multi-stage map-matching pipeline.

**Stage 1 – Data Processing**

GPS tracks, road-network nodes, road arcs and reference routes are loaded and validated. Coordinates are converted into a local metre-based representation for geographical calculations.

**Stage 2 – Candidate Generation**

For every GPS observation, nearby road segments are identified using a spatial search. The final candidate-generation configuration uses a 60-metre search radius and allows up to 24 candidates per observation.

**Stage 3 – Feature Engineering**

A set of 29 features is extracted for candidate road segments. These features describe distance, direction, vehicle movement, neighbouring observations and road-network connectivity.

**Stage 4 – Machine Learning Classification**

Multiple machine learning models are compared. The selected Random Forest model assigns a match probability to each candidate road segment.

**Stage 5 – Route Reconstruction**

An HMM and Viterbi decoder use the candidate scores and transition information to identify a likely sequence of road segments while considering route continuity.

**Stage 6 – Visualisation and Deployment**

The reconstructed route is displayed using geographical visualisations. A FastAPI backend processes prediction requests, while a separate browser-based interface allows users to select trips and view predictions.

## 1.5 System Architecture

The overall system follows the pipeline below:

1. Input GPS trip and road-network data.
2. Data loading, validation and coordinate processing.
3. Candidate road-segment generation.
4. Extraction of 29 candidate features.
5. Random Forest probability scoring.
6. HMM-based sequence decoding using Viterbi.
7. Predicted route generation.
8. API response and geographical visualisation.

The API and user interface are deployed across two laptops. Laptop 1 runs the API, trained model and route decoder. Laptop 2 runs the evaluator interface in a browser and communicates with the API over the network.

## 1.6 Technology Stack

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | NumPy, Pandas |
| Geographic calculations | SciPy |
| Road-network analysis | NetworkX |
| Machine learning | Scikit-learn |
| Bayesian optimisation | Optuna |
| Route reconstruction | Hidden Markov Model and Viterbi algorithm |
| Geographical visualisation | Folium |
| Backend | FastAPI |
| API server | Uvicorn |
| Testing | Pytest and Python unittest |
| Version control | Git and GitHub |

## 1.7 Team Contributions

The project was developed collaboratively, with responsibilities divided across four team members.

| Team member | Primary responsibilities |
|---|---|
| Shriram | Data processing, candidate generation, feature engineering, reference labels, HMM and Viterbi route reconstruction |
| Sharanya | Model training, model comparison, Bayesian hyperparameter optimisation and evaluation |
| Shriya | FastAPI backend, user interface, geographical visualisation and deployment |
| Abdullah | Exploratory data analysis, GPS and road-network quality analysis, API quality assurance and consolidated technical report |

## 1.8 Project Deliverables

The project deliverables include:

- A machine learning-based GPS map-matching pipeline.
- A trained and tuned Random Forest model.
- An HMM and Viterbi-based route reconstruction component.
- Geographical visualisations of GPS tracks and predicted routes.
- A FastAPI backend and browser-based evaluator interface.
- Automated API quality assurance tests.
- A two-laptop deployment workflow.
- A consolidated technical report documenting the methodology, experiments, results and implementation.