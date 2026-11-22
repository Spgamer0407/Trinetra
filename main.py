import argparse
import sys

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
    parser = argparse.ArgumentParser(
        description="TRINETRA: AI-Assisted Digital Authenticity Verification & Forensic Traceability"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Single Image Analysis
    parser_pi = subparsers.add_parser("process-image", help="Analyze an image for deepfake / manipulation artifacts")
    parser_pi.add_argument("image", help="Path to input image file (JPG, PNG, etc.).")
    parser_pi.add_argument(
        "--threshold",
        type=probability_threshold,
        default=0.5,
        help="Fake-probability threshold for a FAKE verdict (default: 0.5).",
    )
    parser_pi.add_argument(
        "--no-explain",
        dest="explain",
        action="store_false",
        help="Disable Grad-CAM explainability heatmap overlay.",
    )
    parser_pi.add_argument(
        "--register",
        action="store_true",
        help="Register fingerprint into forensic integrity ledger.",
    )

    # 2. Forensic Integrity Verification
    parser_vi = subparsers.add_parser("verify-integrity", help="Verify a file's authenticity against the forensic ledger")
    parser_vi.add_argument("file", help="Path to candidate image/video to verify.")

    # 3. Video Analysis
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
        help="Generate and save Grad-CAM explainability heatmaps for processed faces.",
    )

    # 4. Interactive Web Interface
    parser_web = subparsers.add_parser("web", help="Launch the TRINETRA interactive web interface")
    parser_web.add_argument(
        "--port",
        type=positive_int,
        default=8080,
        help="Port to run the web server on (default: 8080).",
    )

    # 5. Pipeline maintenance
    subparsers.add_parser("build-dataset", help="Build processed face dataset from videos")
    subparsers.add_parser("extract-features", help="Extract DenseNet embeddings from processed faces")
    subparsers.add_parser("train", help="Train the Random Forest model")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "process-image":
        from process_image import process_image

        process_image(
            image_path=args.image,
            threshold=args.threshold,
            explain=args.explain,
            register_ledger=args.register,
        )
    elif args.command == "verify-integrity":
        from forensic_fingerprint import verify_integrity

        result = verify_integrity(args.file)
        print("\n=======================================================")
        print("  TRINETRA Forensic Integrity Verification Report")
        print("=======================================================")
        print(f"File Name:         {result['query_fingerprint']['file_name']}")
        print(f"SHA-256 Digest:    {result['query_fingerprint']['sha256']}")
        print(f"Perceptual pHash:  {result['query_fingerprint']['phash']}")
        print(f"Status:            {result['status']}")
        print(f"Result:            {result['message']}")
        if result.get("matched_record"):
            match = result["matched_record"]
            print("\nMatched Ledger Record:")
            print(f"  Reference Name:  {match.get('file_name')}")
            print(f"  Registered At:   {match.get('timestamp')}")
            print(f"  Recorded Verdict:{match.get('verdict', 'N/A')}")
            if result.get("hamming_distance") is not None:
                print(f"  Hamming Distance:{result['hamming_distance']}")
        print("=======================================================\n")
    elif args.command == "web":
        from web_app import run_server

        run_server(port=args.port)
    elif args.command == "process-video":
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
