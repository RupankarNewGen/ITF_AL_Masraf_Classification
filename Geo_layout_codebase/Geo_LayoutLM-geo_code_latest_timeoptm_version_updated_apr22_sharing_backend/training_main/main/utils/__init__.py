import os
import datetime

import torch
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
# from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.strategies import DDPStrategy
from constants import root_path, model_ckpt_path

# num_classes=6
# num_classes=41
#/home/ntlpt-42/Documents/mani_projects/IDP/IDE/Geolayoutlm/geolayoutlm_code_base_2/configs/val_config.yml
def get_config(default_conf_file="./configs/val_config.yml"):
    cfg = OmegaConf.load(default_conf_file)
    cfg_cli = _get_config_from_cli()
    if "config" in cfg_cli:
        cfg_cli_config = OmegaConf.load(cfg_cli.config)
        cfg = OmegaConf.merge(cfg, cfg_cli_config)
        del cfg_cli["config"]

    cfg = OmegaConf.merge(cfg, cfg_cli)

    _update_config(cfg)

    return cfg


def _get_config_from_cli():
    cfg_cli = OmegaConf.from_cli()
    cli_keys = list(cfg_cli.keys())
    for cli_key in cli_keys:
        if "--" in cli_key:
            cfg_cli[cli_key.replace("--", "")] = cfg_cli[cli_key]
            del cfg_cli[cli_key]

    return cfg_cli


def _update_config(cfg):
    with open(os.path.join(root_path, 'label.txt'), "r", encoding="utf-8") as file:
        labels = [line.strip() for line in file]
    num_classes = len(labels)
    # if os.path.exists(cfg.workspace):
    #     cfg.workspace = cfg.workspace.rstrip('/') + '_' + datetime.datetime.now().strftime('%m%d%H%M')
    cfg.workspace = os.path.join(root_path,'results/custom_trial')
    cfg.save_weight_dir = os.path.join(cfg.workspace, "checkpoints")
    print(cfg.save_weight_dir)
    cfg.tensorboard_dir = os.path.join(cfg.workspace, "tensorboard_logs")
    cfg.dataset_root_path = os.path.join(root_path, 'data_in_funsd_format/dataset')
    cfg.model.model_ckpt = model_ckpt_path
    if cfg.dataset == "funsd":
        cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, "funsd_geo")
        cfg.model.n_classes = 7

    elif cfg.dataset == "cord":
        cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, "cord_geo")
        cfg.model.n_classes = 2 * 22 + 1

    elif cfg.dataset == 'custom':
        cfg.dataset_root_path = os.path.join(cfg.dataset_root_path, "custom_geo")
        cfg.model.n_classes = 2 * num_classes+ 1

    # set per-gpu batch size
    num_devices = torch.cuda.device_count()
    if num_devices == 0:
        num_devices = 1
    for mode in ["train", "val"]:
        new_batch_size = cfg[mode].batch_size // num_devices
        cfg[mode].batch_size = new_batch_size


def get_callbacks(cfg):
    callbacks = []
    cb1 = CustomModelCheckpoint(
        dirpath=cfg.save_weight_dir, filename='{epoch}-{f1_labeling:.4f}', monitor="f1_labeling", save_top_k=1, mode='max',
         save_last=False, every_n_epochs=1, save_on_train_epoch_end=False
    )
    cb1.CHECKPOINT_NAME_LAST = "{epoch}-last"
    cb1.FILE_EXTENSION = ".pt"
    callbacks.append(cb1)

    cb2 = CustomModelCheckpoint(
        dirpath=cfg.save_weight_dir, filename='{epoch}-{f1_linking:.4f}', monitor="f1_linking", save_top_k=1, mode='max',
         save_last=False, every_n_epochs=1, save_on_train_epoch_end=False
    )
    cb2.CHECKPOINT_NAME_LAST = "{epoch}-last"
    cb2.FILE_EXTENSION = ".pt"
    callbacks.append(cb2)

    return callbacks


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Save a checkpoint at the end of the training epoch."""
        # as we advance one step at end of training, we use `global_step - 1` to avoid saving duplicates
        # trainer.fit_loop.global_step -= 1
        if (
            not self._should_skip_saving_checkpoint(trainer)
            and self._save_on_train_epoch_end
            and self._every_n_epochs > 0
            and (trainer.current_epoch + 1) % self._every_n_epochs == 0
        ):
            self.save_checkpoint(trainer)
        # trainer.fit_loop.global_step += 1


def get_plugins(cfg):
    plugins = []

    if cfg.train.strategy.type == "ddp":
        plugins.append(DDPPlugin())

    return plugins


def get_loggers(cfg):
    loggers = []

    loggers.append(
        TensorBoardLogger(
            cfg.tensorboard_dir, name="", version="", default_hp_metric=False
        )
    )

    return loggers


def cfg_to_hparams(cfg, hparam_dict, parent_str=""):
    for key, val in cfg.items():
        if isinstance(val, DictConfig):
            hparam_dict = cfg_to_hparams(val, hparam_dict, parent_str + key + "__")
        else:
            hparam_dict[parent_str + key] = str(val)
    return hparam_dict


def get_specific_pl_logger(pl_loggers, logger_type):
    for pl_logger in pl_loggers:
        if isinstance(pl_logger, logger_type):
            return pl_logger
    return None


def get_class_names(dataset_root_path):
    class_names_file = os.path.join(dataset_root_path, "class_names.txt")
    class_names = (
        open(class_names_file, "r", encoding="utf-8").read().strip().split("\n")
    )
    return class_names


def get_label_map(dataset_root_path):
    label_map_file = os.path.join(dataset_root_path, "class_names.txt")
    label_map = {}
    lines = open(label_map_file, "r", encoding="utf-8").readlines()
    for line_idx, line in enumerate(lines):
        label_map[line_idx] = line.strip()
    return label_map
