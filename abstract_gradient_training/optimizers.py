"""Optimizer used by certified training to bound the model updates."""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch

from abstract_gradient_training import interval_arithmetic

if TYPE_CHECKING:
    from abstract_gradient_training import AGTConfig


class SGD:
    """
    A class implementing the bounded SGD update step with optional learning rate decay.
    """

    def __init__(self, config: AGTConfig) -> None:
        self.lr = config.learning_rate
        # get regularisation parameters
        self.l1_reg = config.l1_reg
        self.l2_reg = config.l2_reg
        # If these parameters are left to default, the optimizer will behave like a standard SGD with constant
        # learning rate. If you set these parameters, then the learning rate will decay like
        # lr = max(lr / (1 + sqrt(decay_rate * epoch), lr_min)
        self.lr_decay = config.lr_decay
        self.lr_min = config.lr_min
        self.epoch = 0

    def step(
        self,
        param_l: list[torch.Tensor],
        param_n: list[torch.Tensor],
        param_u: list[torch.Tensor],
        update_n: list[torch.Tensor],
        update_l: list[torch.Tensor],
        update_u: list[torch.Tensor],
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Compute a sound bound on parameters after an SGD update
            param_n = param_n - learning_rate * update_n.

        Args:
            param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
            param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
            param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
            update_n (list[torch.Tensor]): List of the nominal updates of the network [dW1, db1, ..., dWn, dbn].
            update_l (list[torch.Tensor]): List of the lower bound updates of the network [dW1, db1, ..., dWn, dbn].
            update_u (list[torch.Tensor]): List of the upper bound updates of the network [dW1, db1, ..., dWn, dbn].

        Returns:
            tuple: The updated parameter lists [param_l, param_n, param_u].
        """
        # apply regularisation
        param_l, param_n, param_u = l2_update(param_l, param_n, param_u, self.l2_reg)
        param_l, param_n, param_u = l1_update(param_l, param_n, param_u, self.l1_reg)
        lr = self.lr / (1 + self.lr_decay * self.epoch)
        lr = max(lr, self.lr_min)
        for i in range(len(param_n)):
            interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
            # apply the parameter update
            param_n[i] -= lr * update_n[i]
            param_l[i] -= lr * update_u[i]
            param_u[i] -= lr * update_l[i]
            interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
        self.epoch += 1
        return param_l, param_n, param_u


class SGDM:
    """
    A class implementing a bounded SGD optimizer with momentum.
    """

    def __init__(self, config: AGTConfig, momentum: float, dampening: float, nesterov: bool) -> None:
        self.lr = config.learning_rate
        # get regularisation parameters
        self.l1_reg = config.l1_reg
        self.l2_reg = config.l2_reg
        # If these parameters are left to default, the optimizer will behave like a standard SGD with constant
        # learning rate. If you set these parameters, then the learning rate will decay like
        # lr = max(lr / (1 + sqrt(decay_rate * epoch), lr_min)
        self.lr_decay = config.lr_decay
        self.lr_min = config.lr_min
        self.epoch = 0
        # momentum parameters
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov
        if not 0 <= self.momentum < 1:
            raise ValueError("Momentum must be between 0 and 1.")
        if not 0 <= self.dampening < 1:
            raise ValueError("Dampening must be between 0 and 1.")
        if self.nesterov and (self.momentum <= 0 or self.dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")

        if self.l1_reg > 0:
            raise NotImplementedError("L1 regularization is not supported with momentum.")

    @torch.no_grad()
    def step(
        self,
        param_l: list[torch.Tensor],
        param_n: list[torch.Tensor],
        param_u: list[torch.Tensor],
        update_n: list[torch.Tensor],
        update_l: list[torch.Tensor],
        update_u: list[torch.Tensor],
    ):
        """
        Apply a sound SGD with momentum update the parameters and their bounds.

        Args:
            param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
            param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
            param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
            update_n (list[torch.Tensor]): List of the nominal updates of the network [dW1, db1, ..., dWn, dbn].
            update_l (list[torch.Tensor]): List of the lower bound updates of the network [dW1, db1, ..., dWn, dbn].
            update_u (list[torch.Tensor]): List of the upper bound updates of the network [dW1, db1, ..., dWn, dbn].

        Returns:
            tuple: The updated parameter lists [param_l, param_n, param_u].
        """
        # apply l2 regularization
        if self.l2_reg > 0:
            update_l = [g + self.l2_reg * p for g, p in zip(update_l, param_l)]
            update_n = [g + self.l2_reg * p for g, p in zip(update_n, param_n)]
            update_u = [g + self.l2_reg * p for g, p in zip(update_u, param_u)]

        # apply momentum
        if self.momentum != 0:
            # update "velocity" terms
            if self.epoch == 0:
                self.vel_l = update_l
                self.vel_n = update_n
                self.vel_u = update_u
            else:
                self.vel_l = [self.momentum * v + (1 - self.dampening) * g for v, g in zip(self.vel_l, update_l)]
                self.vel_n = [self.momentum * v + (1 - self.dampening) * g for v, g in zip(self.vel_n, update_n)]
                self.vel_u = [self.momentum * v + (1 - self.dampening) * g for v, g in zip(self.vel_u, update_u)]
            # add to the momentum
            if self.nesterov:
                update_l = [g + self.momentum * v for g, v in zip(update_l, self.vel_l)]
                update_n = [g + self.momentum * v for g, v in zip(update_n, self.vel_n)]
                update_u = [g + self.momentum * v for g, v in zip(update_u, self.vel_u)]
            else:
                update_l = self.vel_l
                update_n = self.vel_n
                update_u = self.vel_u

        # check the update
        for vl, vn, vu in zip(update_l, update_n, update_u):
            interval_arithmetic.validate_interval(vl, vu, vn)

        # apply the update
        lr = self.lr / (1 + self.lr_decay * self.epoch)
        lr = max(lr, self.lr_min)
        for i in range(len(param_n)):
            interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
            # apply the parameter update
            param_n[i] -= lr * update_n[i]
            param_l[i] -= lr * update_u[i]
            param_u[i] -= lr * update_l[i]
            interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
        self.epoch += 1
        return param_l, param_n, param_u


def l1_update(
    param_l: list[torch.Tensor], param_n: list[torch.Tensor], param_u: list[torch.Tensor], l1_reg: float
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute a sound bound on the l1 regularisation parameter update
        param_n = param_n - l1_reg * torch.sign(param_n)
    using interval arithmetic.

    Args:
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        l1_reg (float): The l1 regularisation parameter.

    Returns:
        tuple: The updated parameter lists [param_l, param_n, param_u].
    """
    for i in range(len(param_n)):
        interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
        # compute L1 regularisation parameter update
        param_n[i] = param_n[i] - l1_reg * torch.sign(param_n[i])
        # handle edge case where the l1 update causes the bounds to cross zero
        # clamp crossing indices and update non crossing indices as normal
        crossing = (param_l[i] <= 0) & (param_u[i] >= 0)
        param_l[i] = torch.where(
            crossing, torch.clamp(param_l[i] + l1_reg, max=-l1_reg), param_l[i] - l1_reg * torch.sign(param_l[i])
        )
        param_u[i] = torch.where(
            crossing, torch.clamp(param_u[i] - l1_reg, min=l1_reg), param_u[i] - l1_reg * torch.sign(param_u[i])
        )
        interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
    return param_l, param_n, param_u


def l2_update(
    param_l: list[torch.Tensor], param_n: list[torch.Tensor], param_u: list[torch.Tensor], l2_reg: float
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    """
    Compute a sound bound on the l2 regularisation parameter update using interval arithmetic.

    Args:
        param_l (list[torch.Tensor]): List of the lower bound parameters of the network [W1, b1, ..., Wn, bn].
        param_n (list[torch.Tensor]): List of the nominal parameters of the network [W1, b1, ..., Wn, bn].
        param_u (list[torch.Tensor]): List of the upper bound parameters of the network [W1, b1, ..., Wn, bn].
        l1_reg (float): The l1 regularisation parameter.

    Returns:
        tuple: The updated parameter lists [param_l, param_n, param_u].
    """
    assert 0 <= l2_reg <= 1, "l2_reg must be in the range [0, 1]"
    for i in range(len(param_n)):
        interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
        param_n[i] = (1 - l2_reg) * param_n[i]
        param_l[i] = (1 - l2_reg) * param_l[i]
        param_u[i] = (1 - l2_reg) * param_u[i]
        interval_arithmetic.validate_interval(param_l[i], param_u[i], param_n[i])
    return param_l, param_n, param_u
