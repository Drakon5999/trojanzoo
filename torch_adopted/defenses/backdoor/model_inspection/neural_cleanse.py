#!/usr/bin/env python3

import torch
import numpy as np
import os
from collections.abc import Iterable
from trojanzoo.utils.metric import mask_jaccard, normalize_mad
from trojanzoo.utils.logger import MetricLogger
from trojanzoo.utils.tensor import tanh_func
from torch import optim



def format_list(_list: list, _format: str = ':8.3f') -> str:
    return '[' + ', '.join(['{{{}}}'.format(_format).format(a) for a in _list]) + ']'


class NeuralCleanse():
    r"""Neural Cleanse proposed by Bolun Wang and Ben Y. Zhao
    from University of Chicago in IEEE S&P 2019.

    It is a model inspection backdoor defense
    that inherits :class:`trojanvision.defenses.ModelInspection`.
    (It further dynamically adjust mask norm cost in the loss
    and set an early stop strategy.)

    For each class, Neural Cleanse tries to optimize a recovered trigger
    that any input with the trigger attached will be classified to that class.
    If there is an outlier among all potential triggers, it means the model is poisoned.

    See Also:
        * paper: `Neural Cleanse\: Identifying and Mitigating Backdoor Attacks in Neural Networks`_
        * code: https://github.com/bolunwang/backdoor

    Args:
        nc_cost_multiplier (float): Norm loss cost multiplier.
            Defaults to ``1.5``.
        nc_patience (float): Early stop nc_patience.
            Defaults to ``10.0``.
        nc_asr_threshold (float): ASR threshold in cost adjustment.
            Defaults to ``0.99``.
        nc_early_stop_threshold (float): Threshold in early stop check.
            Defaults to ``0.99``.

    Attributes:
        cost_multiplier_up (float): Value to multiply when increasing cost.
            It equals to ``nc_cost_multiplier``.
        cost_multiplier_down (float): Value to divide when decreasing cost.
            It's set as ``nc_cost_multiplier ** 1.5``.

    Attributes:
        init_cost (float): Initial cost of mask norm loss.
        cost (float): Current cost of mask norm loss.

    .. _Neural Cleanse\: Identifying and Mitigating Backdoor Attacks in Neural Networks:
        https://ieeexplore.ieee.org/document/8835365
    """
    name: str = 'neural_cleanse'

    # @classmethod
    # def add_argument(cls, group: argparse._ArgumentGroup):
    #     super().add_argument(group)
    #     group.add_argument('--nc_cost_multiplier', type=float,
    #                        help='norm loss cost multiplier '
    #                        '(default: 1.5)')
    #     group.add_argument('--nc_patience', type=float,
    #                        help='early stop nc_patience '
    #                        '(default: 10.0)')
    #     group.add_argument('--nc_asr_threshold', type=float,
    #                        help='asr threshold in cost adjustment '
    #                        '(default: 0.99)')
    #     group.add_argument('--nc_early_stop_threshold', type=float,
    #                        help='threshold in early stop check. '
    #                        '(default: 0.99)')
        # group.add_argument('--cost', type=float,
        #                    help='cost of mask norm loss '
        #                    '(default: 1e-3)')
    #     return group

    def __init__(self, cost=1e-3, nc_cost_multiplier: float = 1.5, nc_patience: float = 10.0,
                 nc_asr_threshold: float = 0.99,
                 nc_early_stop_threshold: float = 0.99, **kwargs):
        super().__init__(**kwargs)
        self.init_cost = cost
        self.cost_multiplier_up = nc_cost_multiplier
        self.cost_multiplier_down = nc_cost_multiplier ** 1.5
        self.nc_asr_threshold = nc_asr_threshold
        self.nc_early_stop_threshold = nc_early_stop_threshold
        self.nc_patience = nc_patience
        self.early_stop_patience = self.nc_patience * 2
        self.defense_remask_epoch = 10
        self.defense_remask_lr = 0.1

    def detect(self, **kwargs):
        super().detect(**kwargs)

        self.mark_random_pos = self.attack.mark.mark_random_pos
        mark_keys = ['mark', 'mark_height', 'mark_width',
                     'mark_height_offset', 'mark_width_offset',
                     'mark_random_pos', ]
        self.mark_dict = {key: getattr(self.attack.mark, key) for key in mark_keys}
        self.new_dict = {'mark': torch.zeros(self.attack.mark.mark.size(0),
                                             self.attack.mark.data_shape[-2],
                                             self.attack.mark.data_shape[-1],
                                             device=self.attack.mark.mark.device),
                         'mark_height': self.attack.mark.data_shape[-2],
                         'mark_width': self.attack.mark.data_shape[-1],
                         'mark_height_offset': 0,
                         'mark_width_offset': 0,
                         'mark_random_pos': False,
                         }

        for k, v in self.new_dict.items():
            setattr(self.attack.mark, k, v)
        self.attack.mark.mark.zero_()

        mark_list, loss_list, asr_list = self.get_mark_loss_list()
        mask_norms: torch.Tensor = mark_list[:, -1].flatten(start_dim=1).norm(p=1, dim=1)
        mask_norm_list: list[float] = mask_norms.tolist()
        print()
        print('asr           : ' + format_list(asr_list))
        print('mask norms    : ' + format_list(mask_norm_list))
        print('loss          : ' + format_list(loss_list))
        print()
        print('asr MAD       : ' + format_list(normalize_mad(asr_list).tolist()))
        print('mask norm MAD : ' + format_list(normalize_mad(mask_norms).tolist()))
        print('loss MAD      : ' + format_list(normalize_mad(loss_list).tolist()))

        if not self.mark_random_pos:
            self.attack.mark.mark = mark_list[self.attack.target_class]
            select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
            overlap = mask_jaccard(self.attack.mark.get_mask(),
                                   self.real_mask,
                                   select_num=select_num)
            print(f'Jaccard index: {overlap:.3f}')

    def get_mark_loss_list(self, verbose: bool = True,
                           **kwargs) -> tuple[torch.Tensor, list[float], list[float]]:
        r"""Get list of mark, loss, asr of recovered trigger for each class.

        Args:
            verbose (bool): Whether to output jaccard index for each trigger.
                It's also passed to :meth:`optimize_mark()`.
            **kwargs: Keyword arguments passed to :meth:`optimize_mark()`.

        Returns:
            (torch.Tensor, list[float], list[float]):
                list of mark, loss, asr with length ``num_classes``.
        """
        mark_list: list[torch.Tensor] = []
        loss_list: list[float] = []
        asr_list: list[float] = []
        # todo: parallel to avoid for loop
        file_path = os.path.normpath(os.path.join(
            self.folder_path, self.get_filename() + '.npz'))
        for label in range(self.model.num_classes):
            print('Class: ', (label, self.model.num_classes))
            mark, loss = self.optimize_mark(label, verbose=verbose, **kwargs)
            if verbose:
                asr, _ = self.attack.validate_fn(indent=4)
                if not self.mark_random_pos:
                    select_num = self.attack.mark.mark_height * self.attack.mark.mark_width
                    overlap = mask_jaccard(self.attack.mark.get_mask(),
                                           self.real_mask,
                                           select_num=select_num)
                    print(f'Jaccard index: {overlap:.3f}', indent=4)
            else:
                asr, _ = self.model._validate(get_data_fn=self.attack.get_data,
                                              keep_org=False, poison_label=True,
                                              verbose=False)
            mark_list.append(mark)
            loss_list.append(loss)
            asr_list.append(asr)
            np.savez(file_path, mark_list=np.stack([mark.detach().cpu().numpy() for mark in mark_list]),
                     loss_list=np.array(loss_list))
        print()
        print('Defense results saved at: ' + file_path)
        mark_list_tensor = torch.stack(mark_list)
        return mark_list_tensor, loss_list, asr_list

    def loss(self, _input: torch.Tensor, _label: torch.Tensor, target: int,
             trigger_output: None | torch.Tensor = None,
             **kwargs) -> torch.Tensor:
        r"""Loss function to optimize recovered trigger.

        Args:
            _input (torch.Tensor): Clean input tensor
                with shape ``(N, C, H, W)``.
            _label (torch.Tensor): Clean label tensor
                with shape ``(N)``.
            target (int): Target class.
            trigger_output (torch.Tensor):
                Output tensor of input tensor with trigger.
                Defaults to ``None``.

        Returns:
            torch.Tensor: Scalar loss tensor.
        """
        trigger_input = self.attack.add_mark(_input)
        trigger_label = target * torch.ones_like(_label)
        if trigger_output is None:
            trigger_output = self.model(trigger_input, **kwargs)
        return self.model.loss(trigger_input, trigger_label, _output=trigger_output)

    def optimize_mark(self, label: int,
                      loader: Iterable = None,
                      logger_header: str = '',
                      verbose: bool = True,
                      **kwargs) -> tuple[torch.Tensor, float]:
        r"""
        Args:
            label (int): The class label to optimize.
            loader (collections.abc.Iterable):
                Data loader to optimize trigger.
                Defaults to ``self.dataset.loader['train']``.
            logger_header (str): Header string of logger.
                Defaults to ``''``.
            verbose (bool): Whether to use logger for output.
                Defaults to ``True``.
            **kwargs: Keyword arguments passed to :meth:`loss()`.

        Returns:
            (torch.Tensor, torch.Tensor):
                Optimized mark tensor with shape ``(C + 1, H, W)``
                and loss tensor.
        """

        # parameters to update cost
        self.cost_set_counter = 0
        self.cost_up_counter = 0
        self.cost_down_counter = 0
        self.cost_up_flag = False
        self.cost_down_flag = False

        # counter for early stop
        self.early_stop_counter = 0
        self.early_stop_norm_best = float('inf')

        atanh_mark = torch.randn_like(self.attack.mark.mark, requires_grad=True)
        optimizer = optim.Adam([atanh_mark], lr=self.defense_remask_lr, betas=(0.5, 0.9))
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                            T_max=self.defense_remask_epoch)
        optimizer.zero_grad()
        loader = loader or self.dataset.loader['train']

        # best optimization results
        norm_best: float = float('inf')
        mark_best: torch.Tensor = None
        loss_best: float = None

        logger = MetricLogger(indent=4)
        logger.create_meters(loss='{last_value:.3f}',
                             acc='{last_value:.3f}',
                             norm='{last_value:.3f}',
                             entropy='{last_value:.3f}',)
        batch_logger = MetricLogger()
        logger.create_meters(loss=None, acc=None, entropy=None)

        iterator = range(self.defense_remask_epoch)
        if verbose:
            iterator = logger.log_every(iterator, header=logger_header)
        for _ in iterator:
            batch_logger.reset()
            for data in loader:
                self.attack.mark.mark = tanh_func(atanh_mark)    # (c+1, h, w)
                _input, _label = self.model.get_data(data)
                trigger_input = self.attack.add_mark(_input)
                trigger_label = label * torch.ones_like(_label)
                trigger_output = self.model(trigger_input)

                batch_acc = trigger_label.eq(trigger_output.argmax(1)).float().mean()
                batch_entropy = self.loss(_input, _label,
                                          target=label,
                                          trigger_output=trigger_output,
                                          **kwargs)
                batch_norm: torch.Tensor = self.attack.mark.mark[-1].norm(p=1)
                batch_loss = batch_entropy + self.cost * batch_norm

                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                batch_size = _label.size(0)
                batch_logger.update(n=batch_size,
                                    loss=batch_loss.item(),
                                    acc=batch_acc.item(),
                                    entropy=batch_entropy.item())
            lr_scheduler.step()
            self.attack.mark.mark = tanh_func(atanh_mark)    # (c+1, h, w)

            # check to save best mask or not
            loss = batch_logger.meters['loss'].global_avg
            acc = batch_logger.meters['acc'].global_avg
            norm = float(self.attack.mark.mark[-1].norm(p=1))
            entropy = batch_logger.meters['entropy'].global_avg
            if norm < norm_best:
                mark_best = self.attack.mark.mark.detach().clone()
                loss_best = loss
                logger.update(loss=loss, acc=acc, norm=norm, entropy=entropy)

            if self.check_early_stop(loss=loss, acc=acc, norm=norm, entropy=entropy):
                print('early stop')
                break
        atanh_mark.requires_grad_(False)
        self.attack.mark.mark = mark_best
        return mark_best, loss_best

    def check_early_stop(self, acc: float, norm: float, **kwargs) -> bool:
        # update cost
        if self.cost == 0 and acc >= self.nc_asr_threshold:
            self.cost_set_counter += 1
            if self.cost_set_counter >= self.nc_patience:
                self.cost = self.init_cost
                self.cost_up_counter = 0
                self.cost_down_counter = 0
                self.cost_up_flag = False
                self.cost_down_flag = False
                # print(f'initialize cost to {self.cost:.2f}%.2f')
        else:
            self.cost_set_counter = 0

        if acc >= self.nc_asr_threshold:
            self.cost_up_counter += 1
            self.cost_down_counter = 0
        else:
            self.cost_up_counter = 0
            self.cost_down_counter += 1

        if self.cost_up_counter >= self.nc_patience:
            self.cost_up_counter = 0
            # prints(f'up cost from {self.cost:.4f} to {self.cost * self.cost_multiplier_up:.4f}',
            #        indent=4)
            self.cost *= self.cost_multiplier_up
            self.cost_up_flag = True
        elif self.cost_down_counter >= self.nc_patience:
            self.cost_down_counter = 0
            # prints(f'down cost from {self.cost:.4f} to {self.cost / self.cost_multiplier_down:.4f}',
            #        indent=4)
            self.cost /= self.cost_multiplier_down
            self.cost_down_flag = True

        early_stop = False
        # check early stop
        if norm < float('inf'):
            if norm >= self.nc_early_stop_threshold * self.early_stop_norm_best:
                self.early_stop_counter += 1
            else:
                self.early_stop_counter = 0
        self.early_stop_norm_best = min(norm, self.early_stop_norm_best)

        if self.cost_down_flag and self.cost_up_flag and self.early_stop_counter >= self.early_stop_patience:
            early_stop = True

        return early_stop
