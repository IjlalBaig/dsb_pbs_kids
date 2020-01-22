import random
import os
import pandas as pd

from src.dataset import PBSKidsDataset
from src.model import PBSNet
from src.evalutation import qwk
from src.checkpoint import CheckpointHandler
from src.metrics import EpochAverage

# torch
import torch
from torch.functional import F
from torch.utils.data import Subset, DataLoader
from torch.distributions import Beta

from tensorboardX import SummaryWriter

# Ignite
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.utils import convert_tensor
from ignite.metrics import RunningAverage

# Random seeding
random.seed(99)
torch.manual_seed(99)
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed(99)
    torch.cuda.manual_seed_all(99)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(n_epochs=200, batch_sizes=(12, 12), data_dir="./data", log_dir="./log",
          fractions=(0.9, 0.1), workers=2, use_gpu=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    # Create data loaders
    data_fpath = os.path.join(data_dir, "train.csv")
    train_loader, val_loader = _get_data_loaders(fpath=data_fpath, fractions=fractions,
                                                 batch_sizes=batch_sizes, num_workers=workers)

    # Create model and optimizer
    gameplay_len = train_loader.dataset.dataset.data[0].get("gameplay_data").__len__()
    model = PBSNet(in_features=gameplay_len)
    optim = torch.optim.Adam(model.parameters(), lr=5e-3)

    # Create engines
    mixup_sampler = Beta(2.0, 2.0)
    trainer_engine = create_trainer_engine(model, optim, mixup_sampler, device)
    evaluator_engine = create_evaluator_engine(model, optim, device)

    # Init checkpoint handler)
    model_name = model.__class__.__name__
    checkpoint_handler = CheckpointHandler(log_dir, filename_prefix=model_name, n_saved=3)

    # Init summary writer
    writer = SummaryWriter(log_dir=log_dir)

    # Init progress bar
    pbar = ProgressBar()
    metric_names = ["loss_pred", "loss_mixup"]
    pbar.attach(trainer_engine, metric_names=metric_names)

    # Create event handlers
    @trainer_engine.on(Events.STARTED)
    def read_checkpoint(engine):
        checkpoint_dict = checkpoint_handler.load_checkpoint()
        if checkpoint_dict:
            model.load_state_dict(checkpoint_dict.get("model"))
            optim.load_state_dict(checkpoint_dict.get("optim"))
            model.eval()

            engine.state.epoch = checkpoint_dict.get("epoch")
            engine.state.iteration = checkpoint_dict.get("iteration")

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_checkpoint(engine):
        checkpoint_dict = {"model": model.state_dict(),
                           "optim": optim.state_dict(),
                           "epoch": engine.state.epoch,
                           "iteration": engine.state.iteration}

        checkpoint_handler.save_checkpoint(checkpoint_dict)

    @trainer_engine.on(Events.ITERATION_COMPLETED)
    def log_trainer_metrics(engine):
        for key, value in engine.state.metrics.items():
            writer.add_scalar("training/{}".format(key), value, engine.state.iteration)

    @trainer_engine.on(Events.EPOCH_COMPLETED)
    def log_evaluator_metrics(engine):
        evaluator_engine.run(val_loader)
        for key, value in evaluator_engine.state.metrics.items():
            writer.add_scalar("validation/{}".format(key), value, engine.state.epoch)

    @trainer_engine.on(Events.EXCEPTION_RAISED)
    def handle_exception(engine, e):
        writer.close()
        engine.terminate()
        if isinstance(e, KeyboardInterrupt) and (engine.state.iteration > 1):
            import warnings
            warnings.warn('KeyboardInterrupt caught. Exiting gracefully.')
            log_checkpoint(engine)
        else:
            raise e

    # Run session
    trainer_engine.run(train_loader, max_epochs=n_epochs)
    writer.close()


def _get_data_loaders(fpath, fractions=(0.8, 0.2), batch_sizes=(12, 12),
                      num_workers=4, cache_dpath="./data/preprocessed"):
    dataset = PBSKidsDataset(fpath=fpath, mode="train", cache_dpath=cache_dpath)
    train_size = round(len(dataset) * fractions[0] / sum(fractions))
    val_size = round(len(dataset) * fractions[1] / sum(fractions))

    train_set = Subset(dataset, list(range(0, train_size)))
    val_set = Subset(dataset, list(range(train_size, train_size + val_size)))

    train_loader = DataLoader(dataset=train_set, batch_size=batch_sizes[0],
                              shuffle=True, pin_memory=True,
                              num_workers=num_workers)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_sizes[1],
                            shuffle=False, pin_memory=True,
                            num_workers=num_workers)
    return train_loader, val_loader


def _prepare_batch(batch, device=None, non_blocking=False):
    install_id = batch.get("install_id")
    gameplay_data = batch.get("gameplay_data")
    accuracy_data = batch.get("accuracy_data")
    return (install_id,
            convert_tensor(gameplay_data, device=device, non_blocking=non_blocking),
            convert_tensor(accuracy_data, device=device, non_blocking=non_blocking))


def _sample_accuracy_group(accuracy, sample_nan=False):
    if sample_nan:
        idx = torch.isnan(accuracy).max(dim=1)[1].unsqueeze(1)
    else:
        rand_perm = torch.randint_like(accuracy, low=1, high=128)
        val, idx = torch.topk(rand_perm * (accuracy > -1), 1)

    assessment_id = idx
    accuracy_group = accuracy[torch.arange(0, len(idx.squeeze())), idx.squeeze()].view(accuracy.size(0), -1)
    return assessment_id, accuracy_group


def _one_hot_encode(t, n_classes=4):
    code = torch.zeros(t.size(0), n_classes)
    code[torch.arange(code.size(0)).unsqueeze(1), t.long()] = 1.
    return code


def _one_hot_decode(t, n_classes=4):
    return t.max(dim=1)[1]


def create_trainer_engine(model, optim, mixup_sampler, device='cpu', non_blocking=False):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()

        # Preprocess data
        install_id, x, assessment_accuracy = _prepare_batch(batch, device=device, non_blocking=non_blocking)
        assessment_id, accuracy_group = _sample_accuracy_group(assessment_accuracy)
        y = _one_hot_encode(accuracy_group, n_classes=4)

        # Model prediction
        lambda_ = mixup_sampler.sample()
        mixup_shift = random.randint(1, x.size(0))
        y_pred, y_mix_pred = model(x, assessment_id, mixup_shift, lambda_)

        # Compute loss
        loss_pred = F.mse_loss(y_pred, y)
        loss_mixup = F.mse_loss(y_mix_pred, model.mix(y_pred, y_pred.roll(mixup_shift, dims=0), lambda_))

        loss = loss_pred + loss_mixup

        # Backward pass
        optim.zero_grad()
        loss.backward()
        optim.step()

        return {"loss_pred": loss_pred, "loss_mixup": loss_mixup}

    engine = Engine(_update)

    # Add metrics
    RunningAverage(output_transform=lambda x: x["loss_pred"]).attach(engine, "loss_pred")
    RunningAverage(output_transform=lambda x: x["loss_mixup"]).attach(engine, "loss_mixup")

    return engine


def create_evaluator_engine(model, optim, device='cpu', non_blocking=False):
    if device:
        model.to(device)

    def _inference(engine, batch):
        model.train()

        with torch.no_grad():
            # Preprocess data
            install_id, x, assessment_accuracy = _prepare_batch(batch, device=device, non_blocking=non_blocking)
            assessment_id, accuracy_group = _sample_accuracy_group(assessment_accuracy)
            y = _one_hot_encode(accuracy_group, n_classes=4)

            # Model prediction
            y_pred = model(x, assessment_id)

            # Compute metrics
            loss_pred = F.mse_loss(y_pred, y)
            kappa = qwk(_one_hot_decode(y), _one_hot_decode(y_pred))

            return {"loss_pred": loss_pred, "qwk": torch.tensor(kappa)}

    engine = Engine(_inference)

    # Add metrics
    EpochAverage(output_transform=lambda x: x["loss_pred"]).attach(engine, "loss_pred")
    EpochAverage(output_transform=lambda x: x["qwk"]).attach(engine, "qwk")
    return engine


def test(batch_size=12, data_dir="./data", log_dir="./log", workers=2, use_gpu=False):
    device = "cuda" if torch.cuda.is_available() and use_gpu else "cpu"

    # Create data loaders
    data_fpath = os.path.join(data_dir, "test.csv")
    dataset = PBSKidsDataset(fpath=data_fpath, mode="test")
    test_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=workers)

    # Create model
    gameplay_len = dataset.data[0].get("gameplay_data").__len__()
    model = PBSNet(in_features=gameplay_len)

    # Init checkpoint handler
    model_name = model.__class__.__name__
    checkpoint_handler = CheckpointHandler(log_dir, filename_prefix=model_name, n_saved=1)

    # Load model weights
    checkpoint_dict = checkpoint_handler.load_checkpoint()
    model.load_state_dict(checkpoint_dict.get("model"))
    model.eval()

    # Compute Accuracy
    accuracy_groups = []
    for i, batch in enumerate(test_loader):
        install_id, x, assessment_accuracy = _prepare_batch(batch, device=device)
        assessment_id, accuracy_group = _sample_accuracy_group(assessment_accuracy, sample_nan=True)

        y_pred = model(x, assessment_id)
        accuracy_group = _one_hot_decode(y_pred)
        accuracy_groups += accuracy_group.tolist()

    output = pd.read_csv(os.path.join(data_dir, "sample_submission.csv"))
    output["accuracy_group"] = accuracy_groups
    output.to_csv(os.path.join(data_dir, "submission.csv"), index=False)
