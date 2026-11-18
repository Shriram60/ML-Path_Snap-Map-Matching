# 8. Conclusion, Limitations and Future Improvements

## 8.1 Project Conclusion

PathSnap was developed to address the problem of matching noisy GPS observations to the correct road segments and reconstructing a realistic driving route.

The project combines geographical data processing, feature engineering, machine learning, Bayesian hyperparameter optimisation, Hidden Markov Models and geographical visualisation into a complete map-matching application.

The system follows a multi-stage approach:

1. GPS observations are processed and validated.
2. Nearby directed road segments are generated as candidates.
3. A set of 29 engineered features describes the relationship between GPS observations and candidate road segments.
4. A Random Forest classifier estimates the probability that each candidate is the correct match.
5. An HMM with Viterbi decoding reconstructs a consistent sequence of road segments.
6. The FastAPI backend exposes the prediction pipeline to the user interface.
7. Folium provides interactive geographical visualisation of the results.

The Random Forest model was selected following grouped cross-validation and comparison with other machine-learning models. Bayesian optimisation was used to improve the candidate-generation and model configurations. The HMM decoder further improved route consistency.

On the student-test partition, the final system achieved:

| Evaluation metric | Final result |
|---|---:|
| Top-1 Accuracy | 96.74% |
| Route Continuity | 98.18% |
| Reachable Continuity | 99.98% |
| Route F1 Score | 98.67% |

These results demonstrate the effectiveness of combining candidate-level machine-learning predictions with road-network-aware sequence decoding.

The project also produced a working API and completed a two-laptop deployment test, demonstrating that the trained pipeline can be accessed through a separate user interface.

## 8.2 Project Limitations

Although PathSnap achieved strong evaluation results, several limitations were identified during data exploration and development.

### GPS Noise and Irregular Observations

Some trips contain GPS observations that are significantly distant from the reference road network. The EDA identified 22,581 observations more than 50 metres from the nearest reference-route segment.

GPS jitter and large observation gaps can make it difficult to identify the correct road segment.

### Ambiguous Driving Behaviour

Some GPS observations occur while a vehicle is stopped or moving around a turnaround point. In these situations, the driving direction may be unclear, making it difficult to assign a reliable directed road-segment label.

### Candidate Generation Limitations

The prediction pipeline depends on generating the correct road segment as one of the candidates.

If the correct road segment is not included in the candidate set, the classifier and decoder cannot select it. Candidate-generation performance therefore places an upper limit on the final matching performance.

### Road-Network Connectivity

Disconnected road-network transitions and complex road layouts can make route reconstruction challenging.

Although the HMM uses directed road connectivity and bounded shortest-path searches, unusual road-network structures and large GPS jumps can still create difficult cases.

### Dataset Coverage

The project was developed and evaluated using the supplied dataset, consisting of 80 training trips and 10 student-test trips.

Performance on a larger dataset covering additional geographical regions, road types and driving conditions would require further validation.

## 8.3 Future Improvements

Several improvements could be explored in future versions of PathSnap.

### Improved GPS Noise Handling

Additional techniques could be introduced to detect and reduce GPS jitter, identify outliers and handle irregular observation intervals more effectively.

### Better Candidate Generation

Candidate generation could be improved to handle complex road intersections, parallel roads and situations where GPS observations are far from the correct road segment.

### More Advanced Sequence Modelling

Alternative sequence-modelling approaches could be investigated and compared with the existing HMM and Viterbi decoder.

### Expanded Dataset Evaluation

Testing on larger datasets from different cities and geographical regions would help determine how well the system generalises to unfamiliar road networks and driving conditions.

### Production Deployment Improvements

Further work could focus on improving API scalability, concurrent request handling, monitoring, error reporting and deployment reliability.

### Enhanced User Interface

The user interface could be expanded with additional trip statistics, prediction confidence information, route comparison tools and more detailed geographical inspection features.

## 8.4 Final Remarks

PathSnap demonstrates how machine learning and geographical network information can be combined to solve a practical map-matching problem.

The project goes beyond independent GPS-to-road classification by introducing sequence-based route reconstruction, Bayesian optimisation, API integration and interactive visualisation.

The final results show that considering both candidate matching probabilities and road-network connectivity can improve the accuracy and consistency of reconstructed routes.

The project also demonstrates an end-to-end machine-learning workflow, from data exploration and model development to evaluation, application integration and deployment.