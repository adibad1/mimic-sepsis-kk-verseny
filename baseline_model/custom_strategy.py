import os
import joblib
import numpy as np
import flwr as fl
from typing import List, Tuple, Optional, Dict
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from baseline_model.model import get_baseline_model


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        # Call the parent FedAvg class to aggregate the weights from the 5 hospitals
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            print(f"Round {server_round} aggregation successful.")

            # Save the model after the final round (Assuming 5 rounds as configured in server.py)
            if server_round == 5:
                print(
                    "Final round complete. Saving aggregated model weights to disk..."
                )

                # Convert flwr Parameters back to list of numpy arrays
                ndarrays = parameters_to_ndarrays(aggregated_parameters)

                # Instantiate a dummy model and inject the global weights
                global_model = get_baseline_model()
                # Scikit-learn requires classes_ to be set before assigning coefs
                global_model.classes_ = np.array([0, 1])
                global_model.coef_ = ndarrays[0]
                global_model.intercept_ = ndarrays[1]

                # Ensure submission directory exists
                os.makedirs("submission", exist_ok=True)

                # Save the model for the organizer's evaluation script
                save_path = "submission/final_model.pkl"
                joblib.dump(global_model, save_path)
                print(f"Model successfully saved to {save_path}")

        return aggregated_parameters, aggregated_metrics
