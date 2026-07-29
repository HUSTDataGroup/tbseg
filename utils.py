import os
import torch
import logging
import datetime
from collections import defaultdict, deque


def get_timestamp():
    return datetime.datetime.now().strftime("%y%m%d-%H%M%S")


def get_logger(name, root, level=logging.INFO, screen=False, tofile=True):
    os.makedirs(root, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for handler in list(logger.handlers):
            logger.removeHandler(handler)

    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s",
        datefmt="%y-%m-%d %H:%M:%S",
    )

    if tofile:
        log_file = os.path.join(root, name + "_{}.log".format(get_timestamp()))
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if screen:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


class SmoothedValue(object):
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"

        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        if isinstance(value, torch.Tensor):
            value = value.item()

        value = float(value)

        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        if len(self.deque) == 0:
            return 0.0

        values = torch.tensor(list(self.deque), dtype=torch.float32)
        return values.median().item()

    @property
    def avg(self):
        if len(self.deque) == 0:
            return 0.0

        values = torch.tensor(list(self.deque), dtype=torch.float32)
        return values.mean().item()

    @property
    def global_avg(self):
        if self.count == 0:
            return 0.0

        return self.total / self.count

    @property
    def max(self):
        if len(self.deque) == 0:
            return 0.0

        return max(self.deque)

    @property
    def value(self):
        if len(self.deque) == 0:
            return 0.0

        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.item()

            if isinstance(value, (float, int)):
                self.meters[name].update(value)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]

        if attr in self.__dict__:
            return self.__dict__[attr]

        raise AttributeError(
            "'{}' object has no attribute '{}'".format(
                type(self).__name__,
                attr,
            )
        )

    def __str__(self):
        output = []

        for name, meter in self.meters.items():
            output.append("{}: {}".format(name, str(meter)))

        return self.delimiter.join(output)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(
        self,
        step,
        total_steps,
        data_time_value,
        iter_time_value,
        print_freq,
        header=None,
    ):
        if header is None:
            header = ""

        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")

        iter_time.update(iter_time_value)
        data_time.update(data_time_value)

        space_fmt = ":" + str(len(str(total_steps))) + "d"

        if torch.cuda.is_available():
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                    "max mem: {memory:.0f}",
                ]
            )
        else:
            log_msg = self.delimiter.join(
                [
                    header,
                    "[{0" + space_fmt + "}/{1}]",
                    "eta: {eta}",
                    "{meters}",
                    "time: {time}",
                    "data: {data}",
                ]
            )

        if step % print_freq == 0 or step == total_steps - 1:
            eta_seconds = iter_time.global_avg * (total_steps - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

            if torch.cuda.is_available():
                print(
                    log_msg.format(
                        step,
                        total_steps,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
            else:
                print(
                    log_msg.format(
                        step,
                        total_steps,
                        eta=eta_string,
                        meters=str(self),
                        time=str(iter_time),
                        data=str(data_time),
                    )
                )
