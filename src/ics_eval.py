import argparse
import json
import os
from typing import Dict

from .ics_ml import train_or_load_model, evaluate_model


def main():
    parser = argparse.ArgumentParser(description='Train and evaluate CNN models for ICS')
    parser.add_argument('--data-dir', type=str, default='data', help='Dataset root directory')
    parser.add_argument('--out', type=str, default=os.path.join('models', 'metrics'), help='Output directory for metrics JSON files')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs when (re)training')
    parser.add_argument('--force', action='store_true', help='Force retrain even if model file exists')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable automatic class weighting')
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    results: Dict[str, Dict] = {}
    for arch in ['simple', 'deep', 'bn']:
        model_path = os.path.join('models', f'incident_cnn_{arch}.h5')
        print(f"\n=== Training/Evaluating {arch} model ===")
        model = train_or_load_model(
            args.data_dir,
            model_path,
            arch=arch,
            epochs=args.epochs,
            force=args.force,
            use_class_weights=not args.no_class_weights,
        )
        metrics = evaluate_model(model, args.data_dir)
        results[arch] = metrics
        with open(os.path.join(args.out, f'metrics_{arch}.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"{arch} accuracy: {metrics['accuracy']:.3f}")

    # Save a combined summary for your report
    with open(os.path.join(args.out, 'summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved metrics to {args.out}")


if __name__ == '__main__':
    main()
