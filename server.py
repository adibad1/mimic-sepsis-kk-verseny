import flwr as fl
from baseline_model.custom_strategy import SaveModelStrategy


def main():
    print("Starting Central FL Server on port 8080...")

    # ⚠️ CRITICAL RULE: min_fit_clients and min_available_clients MUST be 5.
    # If you alter this, your model will not train on all hospitals and will be disqualified.
    strategy = SaveModelStrategy(
        min_fit_clients=5,
        min_evaluate_clients=5,
        min_available_clients=5,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
    )

    # Start the server on port 8080 to allow hospital clients to connect
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
