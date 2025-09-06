from pathlib import Path
import subprocess
import sys


def get_pending_models():
    """Get list of models that haven't been evaluated yet"""
    models_dir = Path("Models")
    metrics_dir = Path("Testing/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    models = []

    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        metadata_path = model_dir / "training_metadata.json"
        if not metadata_path.exists():
            continue

        metrics_path = metrics_dir / f"{model_dir.name}.json"
        if metrics_path.exists():
            continue

        models.append(str(model_dir))

    return models


def main():
    models = get_pending_models()

    if not models:
        print("No models left to evaluate")
        sys.exit(0)

    print(f"Found {len(models)} models to evaluate:")
    for model in models:
        print(f"- {model}")
    print()

    for model in models:
        print(f"\nEvaluating {model}...")
        try:
            subprocess.run(["python", "Testing/02_TestModels.py", model], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error evaluating {model}: {e}")
            continue
        except KeyboardInterrupt:
            print("\nEvaluation interrupted by user")
            sys.exit(1)


if __name__ == "__main__":
    main()
