from argparse import ArgumentParser


def add_experiment_args(parser: ArgumentParser) -> None:
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--epochs", default=1200, type=int)
    parser.add_argument("--lr_drop", default=300, type=int)
    parser.add_argument("--num_classes", default=4, type=int)

    parser.add_argument("--model", default="TBSeg", type=str)
    parser.add_argument("--dataset", default="MSCMR", type=str)
    parser.add_argument("--dataset_dir", default="./datasets/MSCMR", type=str)

    parser.add_argument("--ce_loss_coef", default=1.0, type=float)
    parser.add_argument("--dice_loss_coef", default=1.0, type=float)
    parser.add_argument("--bayes_loss_coef", default=100.0, type=float)

    parser.add_argument("--visual_interval", default=50, type=int)


def add_management_args(parser: ArgumentParser) -> None:
    parser.add_argument("--output_dir", default="./logs/model", type=str)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_workers", default=4, type=int)


def add_bayes_args(parser: ArgumentParser) -> None:
    parser.add_argument("--phi_rho", default=1e-6, type=float)
    parser.add_argument("--gamma_rho", default=2.0, type=float)

    parser.add_argument("--phi_upsilon", default=1e-8, type=float)
    parser.add_argument("--gamma_upsilon", default=2.0, type=float)

    parser.add_argument("--phi_omega", default=1e-4, type=float)
    parser.add_argument("--gamma_omega", default=2.0, type=float)

    parser.add_argument("--alpha_pi", default=2.0, type=float)
    parser.add_argument("--beta_pi", default=2.0, type=float)

    parser.add_argument("--ds", default=64, type=int)
    parser.add_argument("--nu", default=4.0, type=float)
    parser.add_argument("--gamma_grad", default=10.0, type=float)

    parser.add_argument("--num_feat", default=64, type=int)
    parser.add_argument("--shape_blocks", default=10, type=int)
