#!/usr/bin/env python3

from trojanvision.shortcut.pgd import PGD
from trojanvision.datasets import ImageSet
from trojanvision.utils import apply_cmap
from trojanvision.utils.sgm import register_hook
from trojanzoo.models import _Model, Model
from trojanzoo.environ import env
from trojanzoo.utils.fim import KFAC, EKFAC
from trojanzoo.utils.tensor import add_noise

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.hooks import RemovableHandle
from torchvision.transforms import Normalize
import re

from typing import TYPE_CHECKING
from typing import Union
from trojanzoo.utils.model import ExponentialMovingAverage
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
import argparse
from matplotlib.colors import Colormap  # type: ignore  # TODO
from collections.abc import Callable
if TYPE_CHECKING:
    import torch.utils.data

from matplotlib.cm import get_cmap  # type: ignore  # TODO
jet = get_cmap('jet')

log_softmax = nn.LogSoftmax(1)
criterion_kl = nn.KLDivLoss(reduction='batchmean')


def replace_bn_to_gn(model: nn.Module) -> None:
    r"""Replace all :any:`torch.nn.BatchNorm2d` to :any:`torch.nn.GroupNorm`."""
    for name, module in model.named_children():
        replace_bn_to_gn(module)
        if isinstance(module, nn.BatchNorm2d):
            device = module.weight.device
            gn = nn.GroupNorm(module.num_features,
                              module.num_features,
                              device=device)
            setattr(model, name, gn)


def set_first_layer_channel(model: nn.Module,
                            channel: int = 3,
                            **kwargs) -> None:
    r"""
    Replace the input channel of the first
    :any:`torch.nn.Conv2d` or :any:`torch.nn.Linear`.
    """
    for name, module in model.named_children():
        if len(list(module.children())):
            set_first_layer_channel(module, channel=channel)
        elif isinstance(module, nn.Conv2d):
            if module.in_channels == channel:
                return
            keys = ['out_channels', 'kernel_size', 'bias', 'stride', 'padding']
            args = {key: getattr(module, key) for key in keys}
            args['device'] = module.weight.device
            args.update(kwargs)
            new_conv = nn.Conv2d(in_channels=channel, **args)
            setattr(model, name, new_conv)
        elif isinstance(module, nn.Linear):
            if module.in_features == channel:
                return
            keys = ['out_features', 'bias']
            args = {key: getattr(module, key) for key in keys}
            args['device'] = module.weight.device
            args.update(kwargs)
            new_linear = nn.Linear(in_channels=channel, **args)
            setattr(model, name, new_linear)
        break


class _ImageModel(_Model):
    def __init__(self, norm_par: dict[str, list[float]] = None,
                 num_classes: int = 1000, **kwargs):
        super().__init__(num_classes=num_classes, norm_par=norm_par, **kwargs)

    @classmethod
    def define_preprocess(cls, norm_par: dict[str, list[float]] = None,
                          **kwargs) -> nn.Module:
        if norm_par is not None:
            return Normalize(mean=norm_par['mean'],
                             std=norm_par['std'])
        return super().define_preprocess(**kwargs)


class ImageModel(Model):
    r"""
    | A basic image model wrapper class, which should be the most common interface for users.
    | It inherits :class:`trojanzoo.models.Model` and further extend
      adversarial training and Skip Gradient Method (SGM).

    See Also:
        Adversarial Training:

            * Free: https://github.com/mahyarnajibi/FreeAdversarialTraining
            * Fast: https://github.com/locuslab/fast_adversarial

        Skip Gradient Method: https://github.com/csdongxian/skip-connections-matter

    Attributes:
        adv_train (str | None): choose from ``[None, 'pgd', 'free', 'trades']``.
        adv_train_random_init (bool): Whether to random initialize adversarial noise
            using normal distribution with :attr:`adv_train_eps`.
            Otherwise, attack starts from the benign inputs.
            Defaults to ``False``.
        adv_train_iter (int): Adversarial training PGD iteration.
            Defaults to ``7``.
        adv_train_alpha (float): Adversarial training PGD alpha.
            Defaults to :math:`\frac{2}{255}`.
        adv_train_eps (float): Adversarial training PGD eps.
            Defaults to :math:`\frac{8}{255}`.
        adv_train_eval_iter (int): Adversarial training PGD iteration at evaluation.
            Defaults to :attr:`adv_train_iter`.
        adv_train_eval_alpha (float): Adversarial training PGD alpha at evaluation.
            Defaults to :attr:`adv_train_alpha`.
        adv_train_eval_eps (float): Adversarial training PGD eps at evaluation.
            Defaults to :attr:`adv_train_eps`.
        adv_train_trades_beta (float): regularization factor
            (:math:`\frac{1}{\lambda}` in TRADES)
            Defaults to ``6.0``.
        norm_layer (str): The normalization layer type.
            Choose from ``['bn', 'gn']``.
            Defaults to ``['bn']``.
        sgm (bool): Whether to use Skip Gradient Method. Defaults to ``False``.
        sgm_gamma (float): The gradient factor :math:`\gamma` used in SGM.
            Defaults to ``1.0``.
    """

    @classmethod
    def add_argument(cls, group: argparse._ArgumentGroup):
        r"""Add image model arguments to argument parser group.
        View source to see specific arguments.

        Note:
            This is the implementation of adding arguments.
            The concrete model class may override this method to add more arguments.
            For users, please use :func:`add_argument()` instead, which is more user-friendly.

        See Also:
            :meth:`trojanzoo.models.Model.add_argument()`
        """
        super().add_argument(group)
        group.add_argument('--adv_train', choices=[None, 'pgd', 'free', 'trades'],
                           help='adversarial training.')
        group.add_argument('--adv_train_random_init', action='store_true')
        group.add_argument('--adv_train_iter', type=int,
                           help='adversarial training PGD iteration (default: 7).')
        group.add_argument('--adv_train_alpha', type=float,
                           help='adversarial training PGD alpha (default: 2/255).')
        group.add_argument('--adv_train_eps', type=float,
                           help='adversarial training PGD eps (default: 8/255).')
        group.add_argument('--adv_train_eval_iter', type=int)
        group.add_argument('--adv_train_eval_alpha', type=float)
        group.add_argument('--adv_train_eval_eps', type=float)
        group.add_argument('--adv_train_trades_beta', type=float,
                           help='regularization, i.e., 1/lambda in TRADES '
                           '(default: 6.0)')

        group.add_argument('--norm_layer', choices=['bn', 'gn'], default='bn')
        group.add_argument('--sgm', action='store_true',
                           help='whether to use sgm gradient (default: False)')
        group.add_argument('--sgm_gamma', type=float,
                           help='sgm gamma (default: 1.0)')
        return group

    def __init__(self, name: str = 'imagemodel', layer: int = None,
                 model: Union[type[_ImageModel], _ImageModel] = _ImageModel,
                 dataset: ImageSet = None, data_shape: list[int] = None,
                 adv_train: str = None, adv_train_random_init: bool = False, adv_train_eval_random_init: bool = None,
                 adv_train_iter: int = 7, adv_train_alpha: float = 2 / 255, adv_train_eps: float = 8 / 255,
                 adv_train_eval_iter: int = None, adv_train_eval_alpha: float = None, adv_train_eval_eps: float = None,
                 adv_train_trades_beta: float = 6.0,
                 norm_layer: str = 'bn', sgm: bool = False, sgm_gamma: float = 1.0,
                 norm_par: dict[str, list[float]] = None, suffix: str = None, **kwargs):
        name = self.get_name(name, layer=layer)
        if norm_par is None and isinstance(dataset, ImageSet):
            norm_par = None if dataset.normalize else dataset.norm_par
        if 'num_classes' not in kwargs.keys() and dataset is None:
            kwargs['num_classes'] = 1000
        if adv_train is not None and suffix is None:
            suffix = '_adv_train'
        super().__init__(name=name, model=model, dataset=dataset, data_shape=data_shape,
                         norm_par=norm_par, suffix=suffix, **kwargs)
        assert norm_layer in ['bn', 'gn']
        if norm_layer == 'gn':
            replace_bn_to_gn(self._model)

        if data_shape is None:
            assert isinstance(dataset, ImageSet), 'Please specify data_shape or dataset'
            data_shape = dataset.data_shape
        args = {'padding': 3} if 'vgg' in name else {}  # TODO: so ugly
        set_first_layer_channel(self._model.features,
                                channel=data_shape[0], **args)

        self.sgm: bool = sgm
        self.sgm_gamma: float = sgm_gamma
        self.adv_train = adv_train
        self.adv_train_random_init = adv_train_random_init
        self.adv_train_eval_random_init = adv_train_eval_random_init if adv_train_eval_random_init is not None else adv_train_random_init
        self.adv_train_iter = adv_train_iter
        self.adv_train_alpha = adv_train_alpha
        self.adv_train_eps = adv_train_eps
        self.adv_train_eval_iter = adv_train_eval_iter if adv_train_eval_iter is not None else adv_train_iter
        self.adv_train_eval_alpha = adv_train_eval_alpha if adv_train_eval_alpha is not None else adv_train_alpha
        self.adv_train_eval_eps = adv_train_eval_eps if adv_train_eval_eps is not None else adv_train_eps
        self.adv_train_trades_beta = adv_train_trades_beta

        self.param_list['imagemodel'] = []
        if sgm:
            self.param_list['imagemodel'].append('sgm_gamma')
            register_hook(self, sgm_gamma)
        if adv_train is not None:
            if 'suffix' not in self.param_list['model']:
                self.param_list['model'].append('suffix')
            self.param_list['adv_train'] = ['adv_train', 'adv_train_random_init', 'adv_train_eval_random_init',
                                            'adv_train_iter', 'adv_train_alpha', 'adv_train_eps',
                                            'adv_train_eval_iter', 'adv_train_eval_alpha', 'adv_train_eval_eps']
            if adv_train == 'trades':
                self.param_list['adv_train'].append('adv_train_trades_beta')
            clip_min, clip_max = 0.0, 1.0
            if norm_par is None and isinstance(dataset, ImageSet):
                if dataset.normalize and dataset.norm_par is not None:
                    mean = torch.tensor(dataset.norm_par['mean'],
                                        device=env['device']).view(-1, 1, 1)
                    std = torch.tensor(dataset.norm_par['std'],
                                       device=env['device']).view(-1, 1, 1)
                    clip_min, clip_max = -mean / std, (1 - mean) / std
                    self.adv_train_eval_alpha /= std
                    self.adv_train_eval_eps /= std
                    self.adv_train_alpha /= std
                    self.adv_train_eps /= std
            self.pgd = PGD(pgd_alpha=self.adv_train_eval_alpha, pgd_eps=self.adv_train_eval_eps,
                           iteration=self.adv_train_eval_iter, stop_threshold=None,
                           target_idx=0,
                           random_init=self.adv_train_eval_random_init,
                           clip_min=clip_min, clip_max=clip_max,
                           model=self, dataset=self.dataset)
        self._model: _ImageModel
        self.dataset: ImageSet
        self.sgm_remove: list[RemovableHandle]

    def trades_loss_fn(self, _input: torch.Tensor, org_prob: torch.Tensor, **kwargs):
        return -criterion_kl(log_softmax(self(_input)), org_prob)

    @classmethod
    def get_name(cls, name: str, layer: int = None) -> str:
        full_list = name.split('_')
        partial_name = full_list[0]
        re_list = re.findall(r'\d+|\D+', partial_name)
        if len(re_list) > 1:
            layer = int(re_list[1])
        elif layer is not None:
            partial_name += str(layer)
        full_list[0] = partial_name
        return '_'.join(full_list)

    def get_heatmap(self, _input: torch.Tensor, _label: torch.Tensor,
                    method: str = 'grad_cam', cmap: Colormap = jet) -> torch.Tensor:
        r"""Use colormap :attr:`cmap` to get heatmap tensor of :attr:`_input`
        w.r.t. :attr:`_label` with :attr:`method`.

        Args:
            _input (torch.Tensor): The (batched) input tensor
                with shape ``([N], C, H, W)``.
            _label (torch.Tensor): The (batched) label tensor
                with shape ``([N])``
            method (str): The method to calculate heatmap.
                Choose from ``['grad_cam', 'saliency_map']``.
                Defaults to ``'grad_cam'``.
            cmap (matplotlib.colors.Colormap): The colormap to use.

        Returns:
            torch.Tensor: The heatmap tensor with shape ([N], C, H, W).

        See Also:
            https://keras.io/examples/vision/grad_cam/
        """
        squeeze_flag = False
        if _input.dim() == 3:
            _input = _input.unsqueeze(0)    # (N, C, H, W)
            squeeze_flag = True
        if isinstance(_label, int):
            _label = [_label] * len(_input)
        _label = torch.as_tensor(_label, device=_input.device)
        heatmap = _input    # linting purpose
        if method == 'grad_cam':    # TODO: python 3.10 match
            feats = self._model.get_fm(_input).detach()   # (N, C', H', W')
            feats.requires_grad_()
            _output: torch.Tensor = self._model.pool(feats)   # (N, C', 1, 1)
            _output = self._model.flatten(_output)   # (N, C')
            _output = self._model.classifier(_output)   # (N, num_classes)
            _output = _output.gather(dim=1, index=_label.unsqueeze(1)).sum()
            grad = torch.autograd.grad(_output, feats)[0]   # (N, C', H', W')
            feats.requires_grad_(False)
            weights = grad.mean(dim=-2, keepdim=True).mean(dim=-1, keepdim=True)    # (N, C', 1, 1)
            heatmap = (feats * weights).sum(dim=1, keepdim=True).clamp(0)  # (N, 1, H', W')
            # heatmap.sub_(heatmap.amin(dim=-2, keepdim=True).amin(dim=-1, keepdim=True))
            heatmap.div_(heatmap.amax(dim=-2, keepdim=True).amax(dim=-1, keepdim=True))
            heatmap: torch.Tensor = F.upsample(heatmap, _input.shape[-2:], mode='bilinear')[:, 0]   # (N, H, W)
            # Note that we violate the image order convension (W, H, C)
        elif method == 'saliency_map':
            _input.requires_grad_()
            _output = self(_input).gather(dim=1, index=_label.unsqueeze(1)).sum()
            grad = torch.autograd.grad(_output, _input)[0]   # (N, C, H, W)
            _input.requires_grad_(False)

            heatmap = grad.abs().amax(dim=1)   # (N, H, W)
            heatmap.sub_(heatmap.amin(dim=-2, keepdim=True).amin(dim=-1, keepdim=True))
            heatmap.div_(heatmap.amax(dim=-2, keepdim=True).amax(dim=-1, keepdim=True))
        heatmap = apply_cmap(heatmap.detach().cpu(), cmap)
        return heatmap[0] if squeeze_flag else heatmap

    def get_data(self, data: tuple[torch.Tensor, torch.Tensor],
                 adv_train: bool = False,
                 **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        # In training process, `adv_train` args will not be passed to `get_data`. So it's always `False`.
        _input, _label = super().get_data(data, **kwargs)
        if adv_train:
            assert self.pgd is not None
            adv_x, _ = self.pgd.optimize(_input=_input, target=_label)
            return adv_x, _label
        return _input, _label

    def _validate(self, adv_train: bool = None, **kwargs) -> tuple[float, float]:
        adv_train = adv_train if adv_train is not None else bool(self.adv_train)
        if not adv_train:
            return super()._validate(**kwargs)
        _, clean_acc = super()._validate(print_prefix='Validate Clean', main_tag='valid clean',
                                         adv_train=False, **kwargs)
        _, adv_acc = super()._validate(print_prefix='Validate Adv', main_tag='valid adv',
                                       adv_train=True, **kwargs)
        return adv_acc, clean_acc + adv_acc

    def _train(self, epochs: int, optimizer: Optimizer, lr_scheduler: _LRScheduler = None,
               adv_train: bool = None,
               lr_warmup_epochs: int = 0,
               model_ema: ExponentialMovingAverage = None,
               model_ema_steps: int = 32,
               grad_clip: float = None, pre_conditioner: Union[KFAC, EKFAC] = None,
               print_prefix: str = 'Epoch', start_epoch: int = 0, resume: int = 0,
               validate_interval: int = 10, save: bool = False, amp: bool = False,
               loader_train: torch.utils.data.DataLoader = None,
               loader_valid: torch.utils.data.DataLoader = None,
               epoch_fn: Callable[..., None] = None,
               get_data_fn: Callable[...,
                                     tuple[torch.Tensor, torch.Tensor]] = None,
               loss_fn: Callable[..., torch.Tensor] = None,
               after_loss_fn: Callable[..., None] = None,
               validate_fn: Callable[..., tuple[float, float]] = None,
               save_fn: Callable[..., None] = None, file_path: str = None,
               folder_path: str = None, suffix: str = None,
               writer=None, main_tag: str = 'train', tag: str = '',
               accuracy_fn: Callable[..., list[float]] = None,
               verbose: bool = True, indent: int = 0, **kwargs) -> None:
        adv_train = adv_train if adv_train is not None else bool(self.adv_train)
        if adv_train:
            after_loss_fn_old = after_loss_fn
            if not callable(after_loss_fn) and hasattr(self, 'after_loss_fn'):
                after_loss_fn_old = getattr(self, 'after_loss_fn')
            loss_fn = loss_fn if callable(loss_fn) else self.loss

            def after_loss_fn_new(_input: torch.Tensor, _label: torch.Tensor, _output: torch.Tensor,
                                  loss: torch.Tensor, optimizer: Optimizer, loss_fn: Callable[..., torch.Tensor] = None,
                                  amp: bool = False, scaler: torch.cuda.amp.GradScaler = None, **kwargs):
                optimizer.zero_grad()
                self.zero_grad()
                if pre_conditioner is not None:
                    pre_conditioner.reset()

                if self.adv_train == 'free':
                    noise = self.pgd.init_noise(_input.shape, pgd_eps=self.adv_train_eps,
                                                random_init=self.adv_train_random_init,
                                                device=_input.device)
                    adv_x = add_noise(x=_input, noise=noise, batch=self.pgd.universal,
                                      clip_min=self.pgd.clip_min, clip_max=self.pgd.clip_max)
                    noise.data = self.pgd.valid_noise(adv_x, _input)
                    for m in range(self.adv_train_iter):
                        loss = loss_fn(adv_x, _label)
                        if amp:
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                        else:
                            loss.backward()
                            optimizer.step()
                        optimizer.zero_grad()
                        self.zero_grad()
                        # self.eval()
                        adv_x, _ = self.pgd.optimize(_input=_input, noise=noise, target=_label, iteration=1,
                                                     pgd_alpha=self.adv_train_alpha, pgd_eps=self.adv_train_eps)
                        # self.train()
                        loss = loss_fn(adv_x, _label)
                else:
                    loss = self.adv_loss(_input=_input, _label=_label, loss_fn=loss_fn)

                if amp:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                if callable(after_loss_fn_old):
                    after_loss_fn_old(_input=_input, _label=_label, _output=_output,
                                      loss=loss, optimizer=optimizer, loss_fn=loss_fn,
                                      amp=amp, scaler=scaler, **kwargs)
            after_loss_fn = after_loss_fn_new

        super()._train(epochs=epochs, optimizer=optimizer, lr_scheduler=lr_scheduler,
                       adv_train=adv_train,
                       lr_warmup_epochs=lr_warmup_epochs,
                       model_ema=model_ema, model_ema_steps=model_ema_steps,
                       grad_clip=grad_clip, pre_conditioner=pre_conditioner,
                       print_prefix=print_prefix, start_epoch=start_epoch,
                       resume=resume, validate_interval=validate_interval,
                       save=save, amp=amp,
                       loader_train=loader_train, loader_valid=loader_valid,
                       epoch_fn=epoch_fn, get_data_fn=get_data_fn,
                       loss_fn=loss_fn, after_loss_fn=after_loss_fn,
                       validate_fn=validate_fn,
                       save_fn=save_fn, file_path=file_path,
                       folder_path=folder_path, suffix=suffix,
                       writer=writer, main_tag=main_tag, tag=tag,
                       accuracy_fn=accuracy_fn,
                       verbose=verbose, indent=indent, **kwargs)

    def adv_loss(self, _input: torch.Tensor, _label: torch.Tensor,
                 loss_fn: Callable[..., torch.Tensor] = None, adv_train: str = None) -> torch.Tensor:
        adv_train = adv_train if adv_train is not None else self.adv_train
        loss_fn = loss_fn if callable(loss_fn) else self.loss
        if adv_train == 'trades':
            noise = 1e-3 * torch.randn_like(_input)
            org_prob = self.get_prob(_input)
            adv_x, _ = self.pgd.optimize(_input=_input, noise=noise, target=_label,
                                         iteration=self.adv_train_iter,
                                         pgd_alpha=self.adv_train_alpha,
                                         pgd_eps=self.adv_train_eps,
                                         loss_fn=self.trades_loss_fn,
                                         loss_kwargs={'org_prob': org_prob.detach()})
            adv_x = _input + (adv_x - _input).detach()
            return loss_fn(_input, _label) - self.adv_train_trades_beta * \
                self.trades_loss_fn(_input=adv_x, org_prob=org_prob)
        else:
            adv_x, _ = self.pgd.optimize(_input=_input, target=_label,
                                         iteration=self.adv_train_iter,
                                         pgd_alpha=self.adv_train_alpha,
                                         pgd_eps=self.adv_train_eps,
                                         random_init=self.adv_train_random_init)
            adv_x = _input + (adv_x - _input).detach()
            return loss_fn(adv_x, _label)
