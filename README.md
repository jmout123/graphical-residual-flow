# Graphical Residual Flow

Implementation of a Residual flow that respects the dependency structure between the input variables as specified by an accompanying belief network. For more details on Residual flows, see [here](https://arxiv.org/abs/1906.02735). In short, at each step of the flow, the following transformation is applied to the input:

    y = x + g(x)

where the spectral norm of the weight matricies of neural network $g(\cdot)$ are constrained such that the Lipschitz constant of $g(\cdot)$ is less than one. This ensures the invertibility of the flow.

See the accompanying notebook for further details and examples of usage.

----

To train a flow for density estimation:

```
python ./src/ex_density_estimation with bn=arithmetic num_blocks=6
```

or as an amortized inference artifact:

```
python ./src/ex_inference with bn='binary-tree' num_blocks=10 hidden_dims='[100,100]'
```