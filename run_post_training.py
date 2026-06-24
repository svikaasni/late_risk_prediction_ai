import sys
from src.explainability import compute_and_save_shap
from src.mlops import simulate_production_monitoring

def main():
    print("Step 1: Running SHAP value calculations...")
    try:
        compute_and_save_shap()
        print("SHAP calculations completed successfully.")
    except Exception as e:
        print(f"Error during SHAP calculations: {e}")
        sys.exit(1)

    print("\nStep 2: Simulating production monitoring and drift detection...")
    try:
        report = simulate_production_monitoring()
        print("Drift detection completed successfully.")
        print(f"Drift monitoring status: {report['drift_detected']}")
    except Exception as e:
        print(f"Error during drift simulation: {e}")
        sys.exit(1)

    print("\nAll post-training tasks completed successfully!")

if __name__ == "__main__":
    main()
