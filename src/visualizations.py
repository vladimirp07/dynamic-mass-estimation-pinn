import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def calculate_metrics(y_real, y_pred):
    """
    Computes performance metrics: MAE, RMSE, and R2.
    """
    mae = mean_absolute_error(y_real, y_pred)
    rmse = np.sqrt(mean_squared_error(y_real, y_pred))
    r2 = r2_score(y_real, y_pred)
    return mae, rmse, r2


def print_metrics_report(y_real, y_pred, model_name):
    """
    Prints a formatted report of PINN estimation performance.
    """
    mae, rmse, r2 = calculate_metrics(y_real, y_pred)
    print("\n" + "="*45)
    print(f"REPORT FOR: {model_name.upper()}")
    print("="*45)
    print(f"MAE  (Mean Absolute Error):     {mae:.2f} kg")
    print(f"RMSE (Root Mean Squared Error):  {rmse:.2f} kg")
    print(f"R^2  (Coefficient of Det.):     {r2:.4f}")
    print("="*45 + "\n")
    return mae, rmse, r2


def plot_estimation_vs_real(times, y_real, y_pred, title, color_pred="orange", save_path=None):
    """
    Plots ground truth payloads vs PINN estimated payloads over time.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(times, y_real, label='Real Payload (Ground Truth)', color='blue', linewidth=2, linestyle='--')
    plt.plot(times, y_pred, label='Estimated Payload (PINN)', color=color_pred, linewidth=2, alpha=0.9)
    
    plt.xlabel('Accumulated Time (s)', fontsize=12)
    plt.ylabel('Payload (kg)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def plot_error_distribution(y_real, y_pred, title, color_kde='#2ca02c', save_path=None):
    """
    Plots the error distribution histogram, kernel density estimation (KDE), 
    and indicates mean (mu) and standard deviation (sigma) margins.
    """
    errors = np.array(y_pred) - np.array(y_real)
    errors = errors[~np.isnan(errors)]
    
    mu = np.mean(errors)
    sigma = np.std(errors)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram density and smooth KDE
    sns.histplot(errors, stat="density", bins=30, color='lightgrey', edgecolor='black', alpha=0.7, ax=ax)
    sns.kdeplot(errors, color=color_kde, linewidth=3, ax=ax)
    
    # Statistical markers
    ax.axvline(mu, color='red', linestyle='-', linewidth=2.5, label=rf'$\mu$ (Mean) = {mu:.1f} kg')
    ax.axvline(mu + sigma, color='red', linestyle='--', linewidth=2, alpha=0.8, label=rf'$\pm 1\sigma$ ({sigma:.1f} kg)')
    ax.axvline(mu - sigma, color='red', linestyle='--', linewidth=2, alpha=0.8)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Estimation Error (kg)', fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()
