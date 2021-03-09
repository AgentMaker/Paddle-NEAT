

import paddle

from .activations import identity_activation, tanh_activation
from .cppn import clamp_weights_, create_cppn, get_coord_inputs


class AdaptiveLinearNet:
    def __init__(
        self,
        delta_w_node,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
    ):

        self.delta_w_node = delta_w_node

        self.n_inputs = len(input_coords)
        self.input_coords = paddle.to_tensor(
            [input_coords], dtype="float32"
        ).squeeze()

        self.n_outputs = len(output_coords)
        self.output_coords = paddle.to_tensor(
            [output_coords], dtype="float32"
        ).squeeze()

        self.weight_threshold = weight_threshold
        self.weight_max = weight_max

        self.activation = activation
        self.cppn_activation = cppn_activation

        self.batch_size = batch_size
        self.reset()

    def get_init_weights(self, in_coords, out_coords, w_node):
        (x_out, y_out), (x_in, y_in) = get_coord_inputs(in_coords, out_coords)

        n_in = len(in_coords)
        n_out = len(out_coords)

        zeros = paddle.zeros([n_out, n_in], dtype="float32")

        weights = self.cppn_activation(
            w_node(
                x_out=x_out,
                y_out=y_out,
                x_in=x_in,
                y_in=y_in,
                pre=zeros,
                post=zeros,
                w=zeros,
            )
        )
        clamp_weights_(weights, self.weight_threshold, self.weight_max)

        return weights

    def reset(self):
        with paddle.no_grad():
            self.input_to_output = (
                self.get_init_weights(
                    self.input_coords, self.output_coords, self.delta_w_node
                )
                .unsqueeze(0)
                .expand([self.batch_size, self.n_outputs, self.n_inputs])
            )

            self.w_expressed = self.input_to_output != 0

            self.batched_coords = get_coord_inputs(
                self.input_coords, self.output_coords, batch_size=self.batch_size
            )

    def activate(self, inputs):
        """
        inputs: (batch_size, n_inputs)

        returns: (batch_size, n_outputs)
        """
        with paddle.no_grad():
            inputs = paddle.to_tensor(
                inputs, dtype="float32"
            ).unsqueeze(2)

            outputs = self.activation(self.input_to_output.matmul(inputs))
            input_activs = inputs.transpose([0, 2, 1]).expand(
                [self.batch_size, self.n_outputs, self.n_inputs]
            )
            output_activs = outputs.expand(
                [self.batch_size, self.n_outputs, self.n_inputs]
            )

            (x_out, y_out), (x_in, y_in) = self.batched_coords

            delta_w = self.cppn_activation(
                self.delta_w_node(
                    x_out=x_out,
                    y_out=y_out,
                    x_in=x_in,
                    y_in=y_in,
                    pre=input_activs,
                    post=output_activs,
                    w=self.input_to_output,
                )
            )

            self.delta_w = delta_w
            shape = delta_w.shape
            self.input_to_output = self.input_to_output.reshape([-1])
            self.w_expressed = paddle.to_tensor(self.w_expressed)
            self.w_expressed = self.w_expressed.numpy().reshape(-1).tolist()
            delta_w = delta_w.reshape([-1])
            for _ in range(delta_w.shape[0]):
                if self.w_expressed[_]:
                    self.input_to_output[_] += delta_w[_]
            self.input_to_output = self.input_to_output.reshape(shape)
            clamp_weights_(
                self.input_to_output, weight_threshold=0.0, weight_max=self.weight_max
            )

        return outputs.squeeze(2)

    @staticmethod
    def create(
        genome,
        config,
        input_coords,
        output_coords,
        weight_threshold=0.2,
        weight_max=3.0,
        output_activation=None,
        activation=tanh_activation,
        cppn_activation=identity_activation,
        batch_size=1,
    ):

        nodes = create_cppn(
            genome,
            config,
            ["x_in", "y_in", "x_out", "y_out", "pre", "post", "w"],
            ["delta_w"],
            output_activation=output_activation,
        )

        delta_w_node = nodes[0]

        return AdaptiveLinearNet(
            delta_w_node,
            input_coords,
            output_coords,
            weight_threshold=weight_threshold,
            weight_max=weight_max,
            activation=activation,
            cppn_activation=cppn_activation,
            batch_size=batch_size,
        )
