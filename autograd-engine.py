import math
import random
import os


# verbose display constants
PRETTY_PRINT_ROUND_LIMIT = 5
VERBOSE = os.environ.get("VERBOSE")


class Value:
    def __init__(self, data, _children=(), _op="", label=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f'Value <data={self.data}, op="{self._op}", label="{self.label}", grad={self.grad}>'

    # self + other
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    # self * other
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    # self ** other
    def __pow__(self, other):
        assert isinstance(
            other, (int, float)
        ), "only supporting int and float powers for now"
        out = Value(self.data**other, (self,), f"** {other}")

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad

        out._backward = _backward
        return out

    # other * self
    def __rmul__(self, other):
        return self * other

    # self / other
    def __truediv__(self, other):
        return self * (other**-1)

    # -self
    def __neg__(self):
        return self * -1

    # self - other
    def __sub__(self, other):
        return self + (-other)

    # other + self
    def __radd__(self, other):
        return self + other

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - out.data**2) * out.grad

        out._backward = _backward

        return out

    def expo(self):
        x = self.data
        out = Value(math.exp(x), (self,), "exp")

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def backward(self):
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


class Neuron:
    def __init__(self, nin):
        # self.w = [Value(0.3) for _ in range(nin)]
        # self.b = Value(0.3)
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))

    def __repr__(self):
        return f"Neuron <weights={[round(w.data, PRETTY_PRINT_ROUND_LIMIT) for w in self.w]}, bias={round(self.b.data, PRETTY_PRINT_ROUND_LIMIT)}>"

    def __call__(self, x):
        act = 0
        for xi, wi in zip(self.w, x):
            act += wi * xi
        act += self.b
        out = act.tanh()
        return out

    def parameters(self):
        return self.w + [self.b]


class Layer:
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.neurons = [Neuron(nin) for _ in range(nout)]

        if VERBOSE:
            for neuron in self.neurons:
                print(neuron)

    def __repr__(self):
        return f"Layer <nin={self.nin}, nout={self.nout}>"

    def __call__(self, x):
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs
        # return outs

    def parameters(self):
        params = []
        for neuron in self.neurons:
            ps = neuron.parameters()
            params.extend(ps)

        return params


class MLP:
    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i + 1]) for i in range(len(nouts))]
        if VERBOSE:
            for layer in self.layers:
                print(layer)

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


y = [2.0, 3.0, -1.0]
z = MLP(3, [4, 4, 1])
print(z(y))
# 41 = 3 -> 4(4 + 4 + 4 + 4), 4 -> 4 (5 + 5 + 5 + 5), 4 -> 1 (5)
print(len(z.parameters()))

# xs = [[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]]

# ys = [1.0, -1.0, -1.0, 1.0]
# ypred = [n(x) for x in xs]
# print("[+] ypred ->", ypred)

# loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])
# print("[+] loss ->", loss)

# ROCK -> -1, SCISSORS -> 1, PAPER -> 0
xs = [[-1.0], [0.0], [1.0]]
ys = [0.0, 1.0, -1.0]

n = MLP(1, [4, 4, 1])
print(n(x) for x in xs)

for k in range(1000):
    # forward pass
    ypred = [n(x) for x in xs]
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])

    for p in n.parameters():
        p.grad = 0.0

    # backward pass
    loss.backward()
    # print(n.parameters())

    # update -> (gradient descent)
    for p in n.parameters():
        p.data += -0.01 * p.grad

    print(ypred)
    print(k, loss.data)

print([n(x) for x in xs])
