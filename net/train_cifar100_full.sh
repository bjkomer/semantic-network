#!/usr/bin/env sh

TOOLS=/home/bjkomer/caffe/build/tools

$TOOLS/caffe train \
    --solver=net/cifar100_full_solver.prototxt

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=net/cifar100_full_solver_lr1.prototxt \
    --snapshot=net_output/cifar100_full_iter_60000.solverstate.h5

# reduce learning rate by factor of 10
$TOOLS/caffe train \
    --solver=net/cifar100_full_solver_lr2.prototxt \
    --snapshot=net_output/cifar100_full_iter_65000.solverstate.h5
