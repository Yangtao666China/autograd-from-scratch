import unittest
import numpy as np
from tiny_autograd import Tensor


class GradientTests(unittest.TestCase):
    def check_gradient(self, function, arrays):
        inputs = [Tensor(a, True) for a in arrays]
        function(*inputs).backward()
        for i, a in enumerate(arrays):
            numerical = np.zeros_like(a, dtype=float)
            for index in np.ndindex(a.shape):
                plus, minus = [x.copy() for x in arrays], [x.copy() for x in arrays]
                plus[i][index] += 1e-6
                minus[i][index] -= 1e-6
                numerical[index] = (function(*map(Tensor, plus)).data-function(*map(Tensor, minus)).data)/2e-6
            np.testing.assert_allclose(inputs[i].grad, numerical, atol=1e-6, rtol=1e-5)

    def test_broadcast(self):
        self.check_gradient(lambda x,b: ((x+b)*b).mean(),
                            [np.arange(6.).reshape(2,3)/5, np.array([[.2,.3,.4]])])

    def test_leading_broadcast(self):
        self.check_gradient(lambda x,b: (x*b).sum(), [np.ones((2,3,4)), np.arange(4.)])

    def test_matmul_network(self):
        rng = np.random.default_rng(4)
        self.check_gradient(lambda x,w: (x@w).tanh().softplus().mean(),
                            [rng.normal(size=(3,2)), rng.normal(size=(2,4))])

    def test_unary(self):
        self.check_gradient(lambda x: ((x.exp().log()+x**3)/x).sum(), [np.array([.4,.7,1.2])])

    def test_relu(self):
        self.check_gradient(lambda x: x.relu().sum(), [np.array([-2.,.4,1.2])])

    def test_reduction_axes(self):
        self.check_gradient(lambda x: (x.mean(axis=(0,2))**2).sum(), [np.arange(24.).reshape(2,3,4)/10])

    def test_negative_axis(self):
        self.check_gradient(lambda x: x.mean(axis=-1, keepdims=True).sum(), [np.arange(6.).reshape(2,3)])

    def test_shared_graph_and_repeated_backward(self):
        x = Tensor(3., True)
        y = x*x
        z = y+y+x
        z.backward()
        self.assertEqual(x.grad, 13.)
        z.backward()
        self.assertEqual(x.grad, 26.)
        x.zero_grad()
        z.backward()
        self.assertEqual(x.grad, 13.)

    def test_explicit_gradient(self):
        x = Tensor([2.,3.], True)
        (x*x).backward([1.,2.])
        np.testing.assert_equal(x.grad, [4.,12.])

    def test_stable_softplus(self):
        x = Tensor([-1000.,0.,1000.], True)
        y = x.softplus()
        y.sum().backward()
        self.assertTrue(np.isfinite(y.data).all())
        np.testing.assert_allclose(x.grad, [0.,.5,1.])

    def test_zero_power_at_zero(self):
        x = Tensor(0., True)
        (x**0).backward()
        self.assertEqual(x.grad, 0.)

    def test_deep_graph(self):
        x = Tensor(1., True)
        y = x
        for _ in range(2000):
            y = y+1
        y.backward()
        self.assertEqual(x.grad, 1.)

    def test_invalid_backward(self):
        with self.assertRaises(ValueError):
            Tensor(2.).backward()
        with self.assertRaises(ValueError):
            Tensor([1.,2.], True).backward()
        with self.assertRaises(ValueError):
            Tensor([1.,2.], True).backward([1.])

    def test_constant_not_differentiated(self):
        x, constant = Tensor(2., True), Tensor(3.)
        (x*constant).backward()
        self.assertEqual(constant.grad, 0.)
        self.assertEqual(x.grad, 3.)


if __name__ == "__main__":
    unittest.main()
