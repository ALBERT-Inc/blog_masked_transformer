from datetime import datetime
from pathlib import Path

from ignite.contrib.handlers import ProgressBar
from ignite.handlers import ModelCheckpoint
from ignite.engine import Events, Engine
from ignite.utils import convert_tensor
from ignite.metrics import Average
import torch


def _prepare_batch(batch, device=None, non_blocking=False):
    """Prepare batch for training: pass to a device with options.
    """
    return (convert_tensor(sample, device=device, non_blocking=non_blocking)
            for sample in batch)


def create_densecap_trainer(model, optimizer, loss_fn,
                            device=None, non_blocking=False,
                            prepare_batch=_prepare_batch):
    """
    Factory function for creating a trainer for supervised models.
    Args:
        model (`torch.nn.Module`): the model to train.
        optimizer (`torch.optim.Optimizer`): the optimizer to use.
        loss_fn (torch.nn loss function): the loss function to use.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and
            GPU, the copy may occur asynchronously with respect to the host.
            For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`,
            `device`, `non_blocking` and outputs tuple of tensors
            `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y',
            'y_pred', 'loss' and returns value
            to be assigned to engine's state.output after each iteration.
            Default is returning `loss.item()`.
    Note: `engine.state.output` for this engine is defind by `output_transform`
        parameter and is the loss of the processed batch by default.
    Returns:
        Engine: a trainer engine with supervised update function.
    """
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        rgb, flow, sentence, segments, all_segments = \
            prepare_batch(batch, device=device, non_blocking=non_blocking)
        y_pred = model(rgb, flow, sentence, segments)
        loss = loss_fn(y_pred, (sentence, segments, all_segments))
        loss.backward()
        optimizer.step()
        return loss

    return Engine(_update)


def create_densecap_evaluator(model, metrics=None, device=None,
                              non_blocking=False,
                              prepare_batch=_prepare_batch):
    """
    Factory function for creating an evaluator for supervised models.
    Args:
        model (`torch.nn.Module`): the model to train.
        metrics (dict of str - :class:`~ignite.metrics.Metric`): a map of
            metric names to Metrics.
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
        non_blocking (bool, optional): if True and this copy is between CPU and
            GPU, the copy may occur asynchronously with respect to the host.
            For other cases, this argument has no effect.
        prepare_batch (callable, optional): function that receives `batch`,
            `device`, `non_blocking` and outputs tuple of tensors
            `(batch_x, batch_y)`.
        output_transform (callable, optional): function that receives 'x', 'y',
            'y_pred' and returns value to be assigned to engine's state.output
            after each iteration. Default is returning `(y_pred, y,)` which
            fits output expected by metrics. If you change it you should use
            `output_transform` in metrics.
    Note:
        `engine.state.output` for this engine is defind by `output_transform`
        parameter and is a tuple of `(batch_pred, batch_y)` by default.
    Returns:
        Engine: an evaluator engine with supervised inference function.
    """
    metrics = metrics or {}

    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        model.use_preset('evaluate')
        with torch.no_grad():
            rgb, flow, *targets = \
                prepare_batch(batch, device=device, non_blocking=non_blocking)
            preds = model.predict(rgb, flow)
        return (preds, targets)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)
    return engine


class TrainExtension:
    """Trainer Extension class.
    You can add tranier-extension method (e.g. Registration for TensorBoard,
    Learning Rate Scheduler, etc.) to this class. If you add, then you must
    add such method in also train.py.
    Args:
        res_root_dir (str): result root directory. outputs will be saved in
            ``{res_root_dir}/{task}/{timestamp}/`` directory.
    """
    def __init__(self, trainer, res_dir='results', **kwargs):
        self.trainer = trainer
        self.start_datetime = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self.res_dir = Path(res_dir) / self.start_datetime
        self.res_dir.mkdir(parents=True)

        metric_loss = Average()
        metric_loss.attach(self.trainer, 'loss')

    def set_progressbar(self):
        """Attach ProgressBar.
        Args:
            trainer (ignite.Engine): trainer
            val_evaluator (ignite.Engine): validation evaluator.
            val_evaluator (ignite.Engine): test evaluator.
        """
        pbar = ProgressBar(persist=True)
        pbar.attach(self.trainer)

    def save_model(self, model, save_interval=None, n_saved=1):
        """Extension method for saving model.
        This method saves model as a PyTorch model filetype (.pth). Saved
        file will be saved on `self.res_dir / model / {model_class_name}.pth`.
        Args:
            trainer (ignite.Engine): trainer
            model (torch.nn.Module): model class.
            save_interval (int): Number of epoch interval in which model should
                be kept on disk.
            n_saved (int): Number of objects that should be kept on disk. Older
                files will be removed. If set to None, all objects are kept.
        """
        if save_interval < 0:
            raise ValueError("save_interval must be larger than 0.")
        if n_saved < 0:
            raise ValueError("n_saved must be larger than 0.")
        if isinstance(model, torch.nn.DataParallel):
            model = model.module
        save_handler = ModelCheckpoint(
            self.res_dir / 'model',
            model.__class__.__name__,
            save_interval=save_interval,
            n_saved=n_saved
        )
        self.trainer.add_event_handler(Events.EPOCH_COMPLETED, save_handler,
                                       {'epoch': model})

    def print_metrics(self):
        """Extension method for printing metrics.
        For now, this method prints only validation AP@0.5, mAP@0.5, and
        traning loss.
        Args:
            trainer (ignite.Engine): trainer
            val_evaluator (ignite.Engine): validation evaluator.
        """
        @self.trainer.on(Events.EPOCH_COMPLETED)
        def compute_metrics(engine):
            print(f"Train loss is {self.trainer.state.metrics['loss']}")
