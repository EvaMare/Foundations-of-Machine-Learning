# Exercise Generative Language Modelling

This exercise explores the use of LSTM-networks for the task of poetry generation.
To make it work finish `src/train.py`. The LSTM-cells are typically defined as,

```math
\mathbf{z}_t = \tanh( \mathbf{W}_z \mathbf{x}_t + \mathbf{R}_z \mathbf{h}_{t-1}  + \mathbf{b}_z),
```
```math
\mathbf{i}_t =  \sigma( \mathbf{W}_i \mathbf{x}_t + \mathbf{R}_i \mathbf{h}_{t-1} +\mathbf{b}_i),
```
```math
\mathbf{f}_t = \sigma(\mathbf{W}_f \mathbf{x}_t + \mathbf{R}_f \mathbf{h}_{t-1} + \mathbf{b}_f),
```
```math
\mathbf{c}_t = \mathbf{z}_t \odot \mathbf{i}_t + \mathbf{c}_{t-1} \odot \mathbf{f}_t,
```
```math
\mathbf{o}_t = \sigma(\mathbf{W}_o \mathbf{x}_t + \mathbf{R}_o \mathbf{h}_{t-1} + \mathbf{b}_o),
```
```math
\mathbf{h}_t = \tanh(\mathbf{c}_t) \odot \mathbf{o}_t.
```

The input is denoted as $\mathbf{x}_t \in \mathbb{R}^{n_i}$ and it changes according to time $t$.
Potential new states $\mathbf{z}_t$ are called block input. 
$\mathbf{i}$ is called the input gate. The forget gate is $\mathbf{f}$ and $\mathbf{o}$ denotes the output gate.
$\mathbf{W} \in \mathbb{R}^{n_i \times n_h}$ denotes input,
$\mathbf{R} \in \mathbb{R}^{n_o \times n_h}$ are the recurrent matrices.
$\odot$ indicates element-wise products. 


When you have trained a model, run `src/recurrent_poetry.py`, enjoy!

(Exercise inspired by: https://karpathy.github.io/2015/05/21/rnn-effectiveness/)
