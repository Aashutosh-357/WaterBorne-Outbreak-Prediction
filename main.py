"""
main.py - Entry point for the AquaShield AI Project
===================================================
This script serves as the main entry point to run the data processing 
and model training pipeline.

To start the web dashboard instead, run:
    streamlit run app/app.py
"""

import sys
import os

# Ensure the 'src' directory can be imported
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.model_trainer import full_training_pipeline

if __name__ == "__main__":
    print("="*60)
    print("🌊 AquaShield AI - Starting Core Pipeline")
    print("="*60)
    
    # Run the complete end-to-end pipeline (data preprocessing + training)
    model, metrics = full_training_pipeline(test_size=0.2, save=True)
    
    print("\n✅ Pipeline execution finished successfully!")
    print("You can now launch the dashboard by running: streamlit run app/app.py")
