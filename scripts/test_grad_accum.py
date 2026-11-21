#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Numerical test for the gradient-accumulation change in ltr_trainer.py.

The change is only useful if accumulating N micro-batches of size B produces
the SAME gradient as one batch of size N*B. That is not automatic: forget the
loss/N scaling and the gradient is N times too big (i.e. the learning rate is
silently multiplied by N); zero_grad in the wrong place and you get only the
last micro-batch.

This reproduces the exact code pattern from ltr_trainer.py on a small model
with fixed data, and compares the resulting gradients.

Run:  python test_grad_accum.py
"""

import sys

import torch
import torch.nn as nn


def build():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(16, 32), nn.ReLU(), nn.Linear(32, 4))


def grads_of(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


def big_batch(x, y, loss_fn):
    """Reference: one optimizer step over the whole batch."""
    m = build()
    m.zero_grad(set_to_none=True)
    loss_fn(m(x), y).backward()
    return grads_of(m)


def accumulated(x, y, loss_fn, accum):
    """The pattern used in ltr_trainer.py: loss/accum, step+zero_grad only on
    the accumulation boundary."""
    m = build()
    opt = torch.optim.SGD(m.parameters(), lr=0.0)   # lr=0: we only inspect .grad
    n = x.shape[0] // accum
    total = accum
    for i in range(1, accum + 1):
        xb = x[(i - 1) * n: i * n]
        yb = y[(i - 1) * n: i * n]
        loss = loss_fn(m(xb), yb)
        is_step = (i % accum == 0) or (i == total)
        (loss / accum).backward()
        if is_step:
            g = grads_of(m)          # capture before zero_grad
            opt.step()
            opt.zero_grad(set_to_none=True)
    return g


def wrong_no_scaling(x, y, loss_fn, accum):
    """The classic bug: forgetting to divide by accum."""
    m = build()
    n = x.shape[0] // accum
    for i in range(1, accum + 1):
        loss = loss_fn(m(x[(i - 1) * n: i * n]), y[(i - 1) * n: i * n])
        loss.backward()
    return grads_of(m)


def wrong_zero_each_step(x, y, loss_fn, accum):
    """The other classic bug: zero_grad before every backward."""
    m = build()
    n = x.shape[0] // accum
    for i in range(1, accum + 1):
        m.zero_grad(set_to_none=True)
        (loss_fn(m(x[(i - 1) * n: i * n]), y[(i - 1) * n: i * n]) / accum).backward()
    return grads_of(m)


def maxdiff(a, b):
    return max((a[k] - b[k]).abs().max().item() for k in a)


def main():
    torch.manual_seed(1234)
    B, ACCUM = 32, 2                       # 32 samples, split into 2 x 16
    x = torch.randn(B, 16)
    y = torch.randn(B, 4)
    # mean-reduction loss, same as the tracker losses (l1 / giou / focal all mean)
    loss_fn = nn.MSELoss(reduction="mean")

    ref = big_batch(x, y, loss_fn)
    ok = True

    got = accumulated(x, y, loss_fn, ACCUM)
    d = maxdiff(ref, got)
    print(f"{'[OK] ' if d < 1e-6 else '[FAIL] '}our pattern vs one big batch      "
          f"max|dg| = {d:.3e}   (want ~0)")
    ok &= d < 1e-6

    d = maxdiff(ref, wrong_no_scaling(x, y, loss_fn, ACCUM))
    print(f"{'[OK] ' if d > 1e-3 else '[FAIL] '}bug: no loss/accum scaling       "
          f"max|dg| = {d:.3e}   (want LARGE - proves the test can detect it)")
    ok &= d > 1e-3

    d = maxdiff(ref, wrong_zero_each_step(x, y, loss_fn, ACCUM))
    print(f"{'[OK] ' if d > 1e-3 else '[FAIL] '}bug: zero_grad before each bwd   "
          f"max|dg| = {d:.3e}   (want LARGE - proves the test can detect it)")
    ok &= d > 1e-3

    # accum=4 as well, to make sure it is not a 2-only coincidence
    got4 = accumulated(x, y, loss_fn, 4)
    d = maxdiff(ref, got4)
    print(f"{'[OK] ' if d < 1e-6 else '[FAIL] '}our pattern with accum=4         "
          f"max|dg| = {d:.3e}   (want ~0)")
    ok &= d < 1e-6

    # accum=1 must be identical to the original code path
    got1 = accumulated(x, y, loss_fn, 1)
    d = maxdiff(ref, got1)
    print(f"{'[OK] ' if d < 1e-9 else '[FAIL] '}accum=1 == original behaviour    "
          f"max|dg| = {d:.3e}   (want exactly 0)")
    ok &= d < 1e-9

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
