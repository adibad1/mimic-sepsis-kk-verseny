import flwr as fl
import cloudpickle
import base64
from typing import List, Tuple, Dict
from flwr.common import Metrics
from baseline_model.custom_strategy import SaveModelStrategy
from baseline_model.model import get_model, set_model_parameters


def aggregate_metrics(metrics: List[Tuple[int, Metrics]]) -> Dict[str, float]:
    """
    Aggregates the rich metrics sent by the 5 hospitals.
    Counts (FP, FN, Cost) are summed. Ratios (Accuracy, AUROC) are weighted averaged.
    """
    if not metrics:
        return {}

    total_examples = sum([num_examples for num_examples, _ in metrics])

    # 1. Sum up the absolute counts across all hospitals
    total_cost = sum([m["cost_score"] for _, m in metrics])
    total_fp = sum([m["false_positives"] for _, m in metrics])
    total_fn = sum([m["false_negatives"] for _, m in metrics])
    total_tp = sum([m["true_positives"] for _, m in metrics])
    total_tn = sum([m["true_negatives"] for _, m in metrics])

    # 2. Calculate the weighted average for rates/ratios
    weighted_auroc = sum([num * m["auroc"] for num, m in metrics]) / total_examples
    weighted_log_loss = (
        sum([num * m["log_loss"] for num, m in metrics]) / total_examples
    )
    weighted_accuracy = (
        sum([num * m["accuracy"] for num, m in metrics]) / total_examples
    )
    weighted_f1 = sum([num * m["f1_score"] for num, m in metrics]) / total_examples
    weighted_precision = (
        sum([num * m["precision"] for num, m in metrics]) / total_examples
    )
    weighted_recall = sum([num * m["recall"] for num, m in metrics]) / total_examples

    return {
        "TOTAL_COST": total_cost,
        "FP": total_fp,
        "FN": total_fn,
        "TP": total_tp,
        "TN": total_tn,
        "auroc": weighted_auroc,
        "log_loss": weighted_log_loss,
        "accuracy": weighted_accuracy,
        "f1_score": weighted_f1,
        "precision": weighted_precision,
        "recall": weighted_recall,
    }


def get_on_fit_config_fn():
    model_obj = get_model()
    model_bytes = cloudpickle.dumps(model_obj)
    model_b64 = base64.b64encode(model_bytes).decode("utf-8")

    def fit_config(_server_round: int):
        return {"model_bytes": model_b64}

    return fit_config


def main():
    print("Starting Central FL Server on port 8080...")

    # ⚠️ CRITICAL RULE: min_fit_clients and min_available_clients MUST be 5.
    strategy = SaveModelStrategy(
        min_fit_clients=5,
        min_evaluate_clients=5,  # Ensure all 5 hospitals evaluate the model
        min_available_clients=5,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        on_fit_config_fn=get_on_fit_config_fn(),
        evaluate_metrics_aggregation_fn=aggregate_metrics,  # Use our new smart aggregator
    )

    # Start the server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
