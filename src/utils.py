import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_reconstruction_distribution(error_df, save_path=None):
    """
    Plots the density of reconstruction errors for Normal vs Default.
    This is the KEY plot for your thesis.
    """
    plt.figure(figsize=(10, 6))
    
    sns.kdeplot(data=error_df[error_df['TARGET'] == 0], x='Reconstruction_Error', 
                label='Target 0 (Repayers)', fill=True, color='blue', alpha=0.3)
    
    sns.kdeplot(data=error_df[error_df['TARGET'] == 1], x='Reconstruction_Error', 
                label='Target 1 (Defaulters)', fill=True, color='red', alpha=0.3)
    
    plt.title('Anomaly Detection: Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error (MSE)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"💾 Plot saved to {save_path}")
    plt.show()

def plot_training_loss(history, save_path=None):
    plt.figure(figsize=(8, 4))
    plt.plot(history['loss'], label='Train Loss')
    plt.title('Autoencoder Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    if save_path:
        plt.savefig(save_path)
    plt.show()