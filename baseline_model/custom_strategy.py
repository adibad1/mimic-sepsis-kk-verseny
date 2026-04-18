import joblib
import flwr as fl
from typing import List, Tuple, Optional, Dict
from flwr.common import (
    Parameters,
    FitRes,
    Scalar,
    parameters_to_ndarrays,
)
from baseline_model.model import get_model, set_model_parameters


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            params_numpy = parameters_to_ndarrays(aggregated_parameters)

            full_pipeline = get_model()
            set_model_parameters(full_pipeline, params_numpy)

            joblib.dump(full_pipeline, "final_model.pkl")
            print(f"Round {server_round}: Saved full pipeline to final_model.pkl")

        return aggregated_parameters, aggregated_metrics
