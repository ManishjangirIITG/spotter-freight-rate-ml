from freight_rate.logging_config import setup_logging
from freight_rate.models.grid_search import run_experiment_grid

if __name__ == "__main__":
    setup_logging("grid_search.log")
    best_summary = run_experiment_grid()

    print("\n================ GRID SEARCH COMPLETED ================")
    print(f"Best Run ID:     {best_summary['run_id']}")
    print(f"Best MAE:        ${best_summary['metrics']['mae']:.2f}")
    print(f"Best RMSE:       ${best_summary['metrics']['rmse']:.2f}")
    print(f"Best MAPE:       {best_summary['metrics']['mape']:.2f}%")
    print("Optimal Configurations:")
    for k, v in best_summary["config"].items():
        print(f"  - {k:<25}: {v}")