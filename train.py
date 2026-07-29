import argparse
import datetime
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader

from args import add_bayes_args, add_experiment_args, add_management_args
from data import build_dataset
from models import build_model
from utils import MetricLogger, SmoothedValue, get_logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    cudnn.benchmark = True


def to_float(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().item()

    if isinstance(value, np.generic):
        value = value.item()

    return float(value)


class Trainer:
    def __init__(self, args):
        self.args = args
        self.args.model = "TBSeg"

        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = get_logger(
            name="TBSeg",
            root=self.output_dir,
            screen=True,
            tofile=True,
        )

        self.logger.info(args)

        set_random_seed(args.seed)

        self.device = torch.device(args.device)
        self.writer = SummaryWriter(
            log_dir=os.path.join(self.output_dir, "summary")
        )

        self.model, self.criterion, self.visualizer = build_model(args)
        self.model.to(self.device)

        self.logger.info(self.model)

        n_parameters = sum(
            parameter.numel()
            for parameter in self.model.parameters()
            if parameter.requires_grad
        )

        self.logger.info(
            "number of params: {}".format(n_parameters)
        )

        parameters = [
            parameter
            for parameter in self.model.parameters()
            if parameter.requires_grad
        ]

        self.optimizer = torch.optim.Adam(
            parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer,
            step_size=300,
            gamma=0.1,
        )

        dataset_train = build_dataset(
            image_set="train",
            args=args,
        )

        self.train_loader = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        self.start_epoch = args.start_epoch
        self.epochs = args.epochs
        self.visual_interval = getattr(args, "visual_interval", 50)

        if args.resume:
            self.resume(args.resume)

    def resume(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
        )

        self.model.load_state_dict(checkpoint["model"])

        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        if "lr_scheduler" in checkpoint:
            self.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])

        self.start_epoch = checkpoint.get(
            "epoch",
            self.start_epoch,
        )

        self.logger.info(
            "resumed from epoch {}".format(self.start_epoch)
        )

    def train(self):
        self.logger.info("start training")

        start_time = time.time()

        for epoch in range(self.start_epoch, self.epochs):
            train_stats = self.train_one_epoch(epoch)
            self.lr_scheduler.step()

            self.write_log(epoch, train_stats)
            self.save_checkpoint(epoch + 1)

        self.save_checkpoint(
            self.epochs,
            filename="final_checkpoint.pth",
        )

        total_time = time.time() - start_time
        total_time = str(
            datetime.timedelta(seconds=int(total_time))
        )

        self.logger.info(
            "training time {}".format(total_time)
        )

        self.writer.close()

    def train_one_epoch(self, epoch):
        self.model.train()
        self.criterion.train()

        metric_logger = MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            "lr",
            SmoothedValue(
                window_size=1,
                fmt="{value:.6f}",
            ),
        )

        total_step = len(self.train_loader)

        for step, data_dict in enumerate(self.train_loader):
            start_time = time.time()

            samples = data_dict["image"].to(
                self.device,
                non_blocking=True,
            )

            targets = data_dict["label"].to(
                self.device,
                non_blocking=True,
            )

            data_time = time.time() - start_time

            outputs = self.model(
                samples,
                return_auxiliary=True,
            )

            losses, loss_dict = self.criterion(
                outputs,
                targets,
            )

            if not torch.isfinite(losses):
                print(
                    "loss is {}, stopping training".format(losses)
                )
                print(loss_dict)
                sys.exit(1)

            self.optimizer.zero_grad()
            losses.backward()

            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                1.0,
            )

            self.optimizer.step()

            metric_logger.update(
                loss=losses.item(),
                **loss_dict,
            )

            metric_logger.update(
                lr=self.optimizer.param_groups[0]["lr"]
            )

            iter_time = time.time() - start_time

            metric_logger.log_every(
                step,
                total_step,
                data_time,
                iter_time,
                10,
                "Epoch: [{}]".format(epoch),
            )

            if self.should_visualize(epoch, step):
                self.visualizer(
                    inputs=samples,
                    outputs=outputs["pred_masks"],
                    labels=targets,
                    others=outputs["visualize"],
                    epoch=epoch,
                    writer=self.writer,
                )

        self.logger.info("averaged stats:")
        self.logger.info(metric_logger)

        stats = {
            name: meter.global_avg
            for name, meter in metric_logger.meters.items()
        }

        self.write_tensorboard(epoch, stats)

        return stats

    def should_visualize(self, epoch, step):
        if step != 0:
            return False

        if epoch % self.visual_interval == 0:
            return True

        if epoch + 1 == self.epochs:
            return True

        return False

    def write_tensorboard(self, epoch, stats):
        for name, value in stats.items():
            self.writer.add_scalar(
                "Train/{}".format(name),
                to_float(value),
                epoch,
            )

    def write_log(self, epoch, train_stats):
        log_stats = {
            "epoch": epoch,
            **{
                "train_{}".format(name): to_float(value)
                for name, value in train_stats.items()
            },
        }

        log_path = os.path.join(
            self.output_dir,
            "log.txt",
        )

        with open(log_path, "a", encoding="utf-8") as file:
            file.write(
                json.dumps(log_stats) + "\n"
            )

    def save_checkpoint(self, epoch, filename="checkpoint.pth"):
        checkpoint = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(self.args),
        }

        checkpoint_path = os.path.join(
            self.output_dir,
            filename,
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "TBSeg training",
        allow_abbrev=False,
    )

    add_experiment_args(parser)
    add_management_args(parser)
    add_bayes_args(parser)

    args = parser.parse_args()

    trainer = Trainer(args)
    trainer.train()
