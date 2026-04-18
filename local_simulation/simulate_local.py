import flwr as fl
import numpy as np
import pandas as pd
from baseline_model.model import get_baseline_model
from baseline_model.custom_strategy import SaveModelStrategy


# Mock Client mirroring the hospital clients
class MockClient(fl.client.NumPyClient):
    def __init__(self, hospital_id):
        self.hospital_id = hospital_id

        # Load local dummy data
        df = pd.read_parquet(
            f"local_simulation/dummy_data/hospital_{hospital_id}.parquet"
        )
        df.fillna(df.median(numeric_only=True), inplace=True)  # Mock median imputation
        df.fillna(0, inplace=True)

        self.X = df.drop(columns=["SepsisLabel"]).values
        self.y = df["SepsisLabel"].values

        self.model = get_baseline_model()
        self.model.fit(self.X[:2], self.y[:2])  # Mock initial fit

    def get_parameters(self, config):
        return [self.model.coef_, self.model.intercept_]

    def fit(self, parameters, config):
        self.model.coef_ = parameters[0]
        self.model.intercept_ = parameters[1]
        self.model.fit(self.X, self.y)
        return self.get_parameters(config), len(self.X), {}


def client_fn(cid: str) -> fl.client.Client:
    return MockClient(hospital_id=int(cid) + 1).to_client()


if __name__ == "__main__":
    print("Starting Local Simulation...")
    strategy = SaveModelStrategy(min_fit_clients=5, min_available_clients=5)

    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=5,
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )
