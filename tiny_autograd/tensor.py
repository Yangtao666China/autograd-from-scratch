"""Broadcast-aware autodiff. Matmul deliberately supports 2-D operands only.

Do not mutate data between forward and backward. Leaf gradients accumulate;
call zero_grad before a new optimizer step. Intermediate gradients are reset.
"""

import numpy as np


def _unbroadcast(gradient, shape):
    while gradient.ndim > len(shape):
        gradient = gradient.sum(axis=0)
    for axis, size in enumerate(shape):
        if size == 1:
            gradient = gradient.sum(axis=axis, keepdims=True)
    return gradient.reshape(shape)


class Tensor:
    def __init__(self, data, requires_grad=False, _parents=(), _backward=None):
        self.data = np.array(data, dtype=np.float64, copy=True)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._parents = _parents
        self._backward = _backward or (lambda: None)

    def __repr__(self):
        return f"Tensor({self.data!r}, requires_grad={self.requires_grad})"

    @staticmethod
    def _wrap(value):
        return value if isinstance(value, Tensor) else Tensor(value)

    def _add_grad(self, gradient):
        if self.requires_grad:
            self.grad += _unbroadcast(np.asarray(gradient), self.data.shape)

    def _binary(self, other, forward, left, right):
        other = self._wrap(other)
        out = Tensor(forward(self.data, other.data),
                     self.requires_grad or other.requires_grad, (self, other))

        def backward():
            self._add_grad(left(out.grad, self.data, other.data))
            other._add_grad(right(out.grad, self.data, other.data))

        out._backward = backward
        return out

    def __add__(self, other):
        return self._binary(other, np.add, lambda g, a, b: g, lambda g, a, b: g)

    __radd__ = __add__

    def __mul__(self, other):
        return self._binary(other, np.multiply, lambda g, a, b: g*b, lambda g, a, b: g*a)

    __rmul__ = __mul__

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -self._wrap(other)

    def __rsub__(self, other):
        return self._wrap(other) + -self

    def __truediv__(self, other):
        return self * self._wrap(other)**-1

    def __rtruediv__(self, other):
        return self._wrap(other) * self**-1

    def __pow__(self, exponent):
        if not isinstance(exponent, (int, float)):
            raise TypeError("exponent must be a Python number")
        derivative = (lambda x, y: np.zeros_like(x)) if exponent == 0 else (
            lambda x, y: exponent * x**(exponent-1))
        return self._unary(lambda x: x**exponent, derivative)

    def _unary(self, forward, derivative):
        out = Tensor(forward(self.data), self.requires_grad, (self,))
        out._backward = lambda: self._add_grad(out.grad * derivative(self.data, out.data))
        return out

    def tanh(self):
        return self._unary(np.tanh, lambda x, y: 1-y*y)

    def relu(self):
        return self._unary(lambda x: np.maximum(x, 0), lambda x, y: x > 0)

    def exp(self):
        return self._unary(np.exp, lambda x, y: y)

    def log(self):
        return self._unary(np.log, lambda x, y: 1/x)

    def softplus(self):
        # log(1 + exp(x)), stable even for very large positive logits.
        return self._unary(lambda x: np.logaddexp(0, x),
                           lambda x, y: np.exp(-np.logaddexp(0, -x)))

    def __matmul__(self, other):
        other = self._wrap(other)
        if self.data.ndim != 2 or other.data.ndim != 2:
            raise ValueError("matmul currently requires two matrices")
        out = Tensor(self.data @ other.data, self.requires_grad or other.requires_grad,
                     (self, other))

        def backward():
            self._add_grad(out.grad @ other.data.T)
            other._add_grad(self.data.T @ out.grad)

        out._backward = backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), self.requires_grad, (self,))

        def backward():
            gradient = out.grad
            if axis is not None and not keepdims:
                gradient = np.expand_dims(gradient, axis=axis)
            self._add_grad(np.broadcast_to(gradient, self.data.shape))

        out._backward = backward
        return out

    def mean(self, axis=None, keepdims=False):
        if axis is None:
            count = self.data.size
        else:
            axes = (axis,) if isinstance(axis, int) else axis
            count = np.prod([self.data.shape[a] for a in axes])
        return self.sum(axis=axis, keepdims=keepdims) / float(count)

    def zero_grad(self):
        self.grad.fill(0)

    def backward(self, gradient=None):
        if not self.requires_grad:
            raise ValueError("output does not require gradients")
        if gradient is None:
            if self.data.size != 1:
                raise ValueError("non-scalar output needs an explicit gradient")
            gradient = np.ones_like(self.data)
        gradient = np.asarray(gradient, dtype=np.float64)
        if gradient.shape != self.data.shape:
            raise ValueError("gradient shape must match output shape")
        # Iterative postorder avoids Python's recursion limit for deep graphs.
        order, visited, stack = [], set(), [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
            elif node not in visited:
                visited.add(node)
                stack.append((node, True))
                stack.extend((parent, False) for parent in node._parents)
        for node in order:
            if node._parents:
                node.zero_grad()
        if self._parents:
            self.grad[...] = gradient
        else:
            self.grad += gradient
        for node in reversed(order):
            node._backward()
