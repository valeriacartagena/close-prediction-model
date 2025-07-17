import numpy as np
import matplotlib.pyplot as plt
import json
import os

def backtest_model(preds, actuals, model_name, threshold=0.0, confidence_weighted=False, results_dir='website_results'):
    """
    Unified backtest for any model. Computes and saves PnL, drawdown, metrics, and plot.
    Args:
        preds: Model predictions (array-like)
        actuals: Actual returns (array-like)
        model_name: String for labeling
        threshold: Only take positions where abs(pred) > threshold
        confidence_weighted: If True, use preds * actuals for PnL
        results_dir: Where to save plots and metrics
    Returns:
        dict of metrics
    """
    preds = np.asarray(preds)
    actuals = np.asarray(actuals)
    if confidence_weighted:
        mask = np.abs(preds) > threshold
        pnl_series = preds[mask] * actuals[mask]
        positions = preds[mask]
        y_used = actuals[mask]
    else:
        positions = np.sign(preds)
        mask = np.abs(preds) > threshold
        positions = np.where(mask, positions, 0)
        pnl_series = positions * actuals
        y_used = actuals
    cumulative_pnl = np.cumsum(pnl_series)
    rolling_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = rolling_max - cumulative_pnl
    sharpe = np.mean(pnl_series) / (np.std(pnl_series) + 1e-6)
    hit_rate = np.mean(np.sign(positions) == np.sign(y_used))
    total_pnl = cumulative_pnl[-1] if len(cumulative_pnl) > 0 else 0.0
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0.0
    num_trades = np.count_nonzero(positions)
    # Log shapes and sample values
    print(f"[{model_name}] preds shape: {preds.shape}, sample: {preds[:5]}")
    print(f"[{model_name}] actuals shape: {actuals.shape}, sample: {actuals[:5]}")
    print(f"[{model_name}] positions shape: {positions.shape}, sample: {positions[:5]}")
    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(cumulative_pnl, label="Cumulative PnL")
    plt.plot(drawdown, label="Drawdown")
    plt.title(f"{model_name} Backtest")
    plt.legend()
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_backtest.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    # Save metrics
    metrics = {
        "sharpe": float(sharpe),
        "hit_rate": float(hit_rate),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(max_drawdown),
        "num_trades": int(num_trades),
        "threshold": float(threshold),
        "confidence_weighted": confidence_weighted
    }
    metrics_path = os.path.join(results_dir, f"{model_name.lower().replace(' ', '_')}_backtest.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics 