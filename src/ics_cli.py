import argparse
import os
from typing import List

from .ics_graph import build_graph, k_shortest_paths
from .ics_map import render_map

# ML is optional at runtime; we'll import lazily when image is provided

def main():
    parser = argparse.ArgumentParser(description='Incident Classification System (ICS) — Assignment 2B')
    parser.add_argument('--origin', type=int, default=1, help='Start node id (default: 1)')
    parser.add_argument('--dest', type=int, nargs='+', default=[10, 13], help='Destination node id(s)')
    parser.add_argument('--k', type=int, default=5, help='Number of top routes to show')
    parser.add_argument('--incident-image', type=str, required=True, help='Path to an incident image for severity classification (REQUIRED)')
    parser.add_argument('--data-dir', type=str, default=os.path.join('data'), help='Dataset root for training/validation')
    parser.add_argument('--model', type=str, default=os.path.join('models', 'incident_cnn.h5'), help='Path to save/load CNN model')
    parser.add_argument('--model-arch', type=str, choices=['simple','deep','bn'], default='simple', help='CNN architecture to use (simple, deep, or bn for batchnorm)')
    parser.add_argument('--epochs', type=int, default=5, help='Training epochs when (re)training the model')
    parser.add_argument('--force-train', action='store_true', help='Force retraining even if model exists')
    parser.add_argument('--no-class-weights', action='store_true', help='Disable class weighting (imbalance handling)')
    parser.add_argument('--output', type=str, default=os.path.join('maps', 'ics_map.html'), help='Output HTML path')
    parser.add_argument('--use-distance-time', action='store_true', help='Compute edge times from distance at a given speed instead of static times')
    parser.add_argument('--speed-kmh', type=float, default=60.0, help='Speed in km/h when using distance-based times (default: 60)')
    parser.add_argument('--algorithm', type=str, choices=['ksp','ucs'], default='ksp', help='Routing algorithm: ksp (NetworkX k-shortest) or ucs (Uniform Cost Search, single path)')
    parser.add_argument('--open', dest='auto_open', action='store_true', help='Open the generated map in the default browser')
    parser.add_argument('--classify-only', action='store_true', help='Only run ML classification and print result; skip map generation')
    args = parser.parse_args()

    # Train or load CNN and predict severity (mandatory for Assignment 2B)
    try:
        from .ics_ml import train_or_load_model, predict_severity
        if not os.path.isfile(args.incident_image):
            raise FileNotFoundError(f"Incident image not found: {args.incident_image}")
        os.makedirs(os.path.dirname(args.model), exist_ok=True)
        model = train_or_load_model(
            args.data_dir,
            args.model,
            arch=args.model_arch,
            epochs=args.epochs,
            force=args.force_train,
            use_class_weights=not args.no_class_weights,
        )
        severity, conf = predict_severity(model, args.incident_image)
        print(f"Predicted incident severity: {severity} (confidence {conf:.2f})")
        # Optional: quick evaluation on validation set to print accuracy
        try:
            from .ics_ml import evaluate_model
            metrics = evaluate_model(model, args.data_dir)
            print(f"Validation accuracy: {metrics['accuracy']:.3f}")
        except Exception:
            pass
        if args.classify_only:
            return
    except Exception as e:
        print("ERROR: ML classification is required for this run.")
        print(str(e))
        raise SystemExit(2)

    G = build_graph(severity, use_distance_time=args.use_distance_time, speed_kmh=args.speed_kmh)

    all_routes: List[List[int]] = []
    if args.algorithm == 'ucs':
        # Use our own UCS implementation for a single best path per destination
        from .search import uniform_cost_search
        for d in args.dest:
            path, cost = uniform_cost_search(G, args.origin, d, weight='weight')
            if path:
                all_routes.append(path)
                print(f"UCS best path to {d}: {cost:.1f} min")
            else:
                print(f"No route found from {args.origin} to {d} using UCS.")
    else:
        for d in args.dest:
            routes = k_shortest_paths(G, args.origin, d, k=args.k)
            if routes:
                all_routes.extend(routes)
                print(f"Found {len(routes)} routes for destination {d}.")
            else:
                print(f"No route found from {args.origin} to {d}.")

    if not all_routes:
        print("No routes to display. Exiting.")
        return

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    # Build an incident panel with embedded image and prediction
    incident_panel_html = None
    try:
        import base64
        mime = 'image/jpeg'
        with open(args.incident_image, 'rb') as f:
            b64 = base64.b64encode(f.read()).decode('ascii')
        img_tag = f"<img src='data:{mime};base64,{b64}' style='max-width:200px; height:auto; display:block; border:1px solid #333; border-radius:4px;'/>"
        incident_panel_html = f"""
        <div style='position: fixed; top: 20px; right: 20px; z-index: 9999; background: rgba(255,255,255,0.95); padding: 10px 12px; border: 1px solid #333; border-radius: 4px; font-size: 12px; line-height: 1.4; max-width: 240px;'>
          <b>Incident classification</b><br>
          {img_tag}
          <div style='margin-top:6px;'>Severity: <b>{severity}</b><br>Confidence: <b>{conf:.2f}</b><br>Model: <b>{args.model_arch}</b></div>
        </div>
        """
    except Exception:
        pass

    # For UCS we show one route per destination; for KSP cap at k
    routes_to_show = all_routes if args.algorithm == 'ucs' else all_routes[:args.k]
    render_map(G, routes_to_show, severity, args.output, incident_panel_html=incident_panel_html)
    abs_out = os.path.abspath(args.output)
    print(f"Map saved to {abs_out} — open this file in a browser.")
    if args.auto_open:
        try:
            import webbrowser
            webbrowser.open('file://' + abs_out)
        except Exception as e:
            print("Could not auto-open the map:", e)


if __name__ == '__main__':
    main()
