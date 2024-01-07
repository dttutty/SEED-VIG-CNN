import wandb

from train import build_parser, train


def main() -> None:
    parser = build_parser("Train the SEED-VIG fusion model with W&B tracking")
    parser.add_argument("--wandb-project", default="seed-vig-cnn")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"))
    args = parser.parse_args()

    config = {
        key: str(value) if key in {"data_dir", "output"} else value
        for key, value in vars(args).items()
        if not key.startswith("wandb_")
    }
    with wandb.init(
        project=args.wandb_project, mode=args.wandb_mode, config=config
    ) as run:
        train(args, on_epoch_end=lambda epoch, metrics: run.log(metrics, step=epoch))
        run.save(str(args.output), policy="now")


if __name__ == "__main__":
    main()
