from keras.layers.merge import _Merge
import keras.backend as K

class NegExpManhattan(_Merge):
    """Layer that computes a dot product between samples in two tensors.
    E.g. if applied to two tensors `a` and `b` of shape `(batch_size, n)`,
    the output will be a tensor of shape `(batch_size, 1)`
    where each entry `i` will be the dot product between
    `a[i]` and `b[i]`.
    # Arguments
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    """

    def __init__(self, axes, **kwargs):
        super(NegExpManhattan, self).__init__(**kwargs)
        if not isinstance(axes, int):
            if not isinstance(axes, (list, tuple)):
                raise TypeError('Invalid type for `axes` - '
                                'should be a list or an int.')
            if len(axes) != 2:
                raise ValueError('Invalid format for `axes` - '
                                 'should contain two elements.')
            if not isinstance(axes[0], int) or not isinstance(axes[1], int):
                raise ValueError('Invalid format for `axes` - '
                                 'list elements should be "int".')
        self.axes = axes
        self.supports_masking = True

    def build(self, input_shape):
        # Used purely for shape validation.
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = input_shape[0]
        shape2 = input_shape[1]
        if shape1 is None or shape2 is None:
            return
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        if shape1[axes[0]] != shape2[axes[1]]:
            raise ValueError(
                'Dimension incompatibility '
                '%s != %s. ' % (shape1[axes[0]], shape2[axes[1]]) +
                'Layer shapes: %s, %s' % (shape1, shape2))

    def call(self, inputs):
        x1 = inputs[0]
        x2 = inputs[1]
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % K.ndim(x1), self.axes % K.ndim(x2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = []
            for i in range(len(self.axes)):
                if self.axes[i] < 0:
                    axes.append(self.axes[i] % K.ndim(inputs[i]))
                else:
                    axes.append(self.axes[i])

        output = K.exp(-1.0*(K.sum(K.abs(x1 - x2), axes)))
        return output

    def compute_output_shape(self, input_shape):
        if not isinstance(input_shape, list) or len(input_shape) != 2:
            raise ValueError('A `Dot` layer should be called '
                             'on a list of 2 inputs.')
        shape1 = list(input_shape[0])
        shape2 = list(input_shape[1])
        if isinstance(self.axes, int):
            if self.axes < 0:
                axes = [self.axes % len(shape1), self.axes % len(shape2)]
            else:
                axes = [self.axes] * 2
        else:
            axes = self.axes
        shape1.pop(axes[0])
        shape2.pop(axes[1])
        shape2.pop(0)
        output_shape = shape1 + shape2
        if len(output_shape) == 1:
            output_shape += [1]
        return tuple(output_shape)

    def compute_mask(self, inputs, mask=None):
        return None

    def get_config(self):
        config = {
            'axes': self.axes
        }
        base_config = super(NegExpManhattan, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def negexp_manhattan(inputs, axes, **kwargs):
    """Functional interface to the `Dot` layer.
    # Arguments
        inputs: A list of input tensors (at least 2).
        axes: Integer or tuple of integers,
            axis or axes along which to take the dot product.
        normalize: Whether to L2-normalize samples along the
            dot product axis before taking the dot product.
            If set to True, then the output of the dot product
            is the cosine proximity between the two samples.
        **kwargs: Standard layer keyword arguments.
    # Returns
        A tensor, the dot product of the samples from the inputs.
    """
    return NegExpManhattan(axes=axes, **kwargs)(inputs)


class AbsDiff(_Merge):
    """Layer that returns the absolute difference (element-wise) of a list of inputs.

    It takes as input a list of tensors,
    all of the same shape, and returns
    a single tensor (also of the same shape).
    """

    def _merge_function(self, inputs):
        output = inputs[0]
        for i in range(1, len(inputs)):
            output -= inputs[i]
        return K.abs(output)


def absdiff(inputs, **kwargs):
    """Functional interface to the `AbsDiff` layer.

    # Arguments
        inputs: A list of input tensors (at least 2).
        **kwargs: Standard layer keyword arguments.

    # Returns
        A tensor, the element-wise product of the inputs.
    """
    return AbsDiff(**kwargs)(inputs)
