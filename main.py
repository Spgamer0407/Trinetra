import argparse

from config import DEFAULT_NUM_FRAMES


def positive_int(value: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def probability_threshold(value: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1") from exc
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("must be a number between 0 and 1")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Trinetra: deepfake face processing pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_pv = subparsers.add_parser("process-video", help="Classify a video using its detected faces")
    parser_pv.add_argument(
        "video",
        nargs="?",
        help="Video file to classify (defaults to sample.mp4 in the project root).",
    )
    parser_pv.add_argument(
        "--frames",
        type=positive_int,
        default=DEFAULT_NUM_FRAMES,
        help=f"Maximum number of frames to sample (default: {DEFAULT_NUM_FRAMES}).",
    )
    parser_pv.add_argument(
        "--threshold",
        type=probability_threshold,
        default=0.5,
        help="Mean fake-probability threshold for a FAKE verdict (default: 0.5).",
    )
    parser_pv.add_argument(
        "--explain",
        action="store_true",
        help="Generate and save Grad-CAM explainability heatmaps for processed faces."
    )

    subparsers.add_parser("build-dataset", help="Build processed face dataset")
    subparsers.add_parser("extract-features", help="Extract DenseNet embeddings")
    subparsers.add_parser("train", help="Train the Random Forest model")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process-video":
        from process_video import main as process_video_main

        process_video_main(
            explain=args.explain,
            video_path=args.video,
            num_frames=args.frames,
            threshold=args.threshold,
        )
    elif args.command == "build-dataset":
        from build_dataset import main as build_dataset_main

        build_dataset_main()
    elif args.command == "extract-features":
        from extract_features import main as extract_features_main

        extract_features_main()
    elif args.command == "train":
        from train_classifier import main as train_classifier_main

        train_classifier_main()


if __name__ == "__main__":
    main()