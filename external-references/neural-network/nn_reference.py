"""
External reference for neural-network DES demos.

This is intentionally dependency-free Python source. It does not provide or
download a solver binary. The TypeScript validator invokes it with python3
(or PYTHON_BIN) and compares:

  1. XOR supervised learning with the same MLP, PRNG, and sample order.
  2. Corridor MDP optimal policy by value iteration.
  3. Neural ODE y' = -rate*y by RK4 against the analytical solution.

Usage:
    python3 external-references/neural-network/nn_reference.py --out out/external/neural-network/reference.json
"""

import argparse
import json
import math
import pathlib
import sys


XOR = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]


class Mulberry32:
    def __init__(self, seed):
        self.s = int(seed) & 0xFFFFFFFF

    def random(self):
        self.s = (self.s + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.s
        t = imul(t ^ unsigned_right_shift(t, 15), t | 1)
        t = to_int32(t ^ to_int32(t + imul(t ^ unsigned_right_shift(t, 7), t | 61)))
        return ((t ^ unsigned_right_shift(t, 14)) & 0xFFFFFFFF) / 4294967296.0


def imul(a, b):
    return to_int32((to_int32(a) * to_int32(b)) & 0xFFFFFFFF)


def to_int32(x):
    x = int(x) & 0xFFFFFFFF
    return x if x < 0x80000000 else x - 0x100000000


def unsigned_right_shift(x, n):
    return (int(x) & 0xFFFFFFFF) >> n


def activate(name, z):
    if name == "linear":
        return z
    if name == "sigmoid":
        return 1.0 / (1.0 + math.exp(-z))
    if name == "tanh":
        return math.tanh(z)
    if name == "relu":
        return z if z > 0 else 0.0
    raise ValueError("unknown activation: " + name)


def activation_prime(name, a, z):
    if name == "linear":
        return 1.0
    if name == "sigmoid":
        return a * (1.0 - a)
    if name == "tanh":
        return 1.0 - a * a
    if name == "relu":
        return 1.0 if z > 0 else 0.0
    raise ValueError("unknown activation: " + name)


class FeedForwardNetwork:
    def __init__(self, layers):
        self.layers = [
            {
                "weights": [list(row) for row in layer["weights"]],
                "biases": list(layer["biases"]),
                "activation": layer["activation"],
            }
            for layer in layers
        ]
        self.input_dim = len(self.layers[0]["weights"][0])
        self.output_dim = len(self.layers[-1]["biases"])

    @staticmethod
    def random(input_dim, hidden_layers, output_dim, hidden_activation, output_activation, rng, weight_scale=None):
        dims = [input_dim] + list(hidden_layers) + [output_dim]
        layers = []
        for k in range(len(dims) - 1):
            fan_in = dims[k]
            fan_out = dims[k + 1]
            limit = weight_scale if weight_scale is not None else math.sqrt(6.0 / (fan_in + fan_out))
            weights = []
            for _ in range(fan_out):
                row = []
                for _ in range(fan_in):
                    row.append((2.0 * rng.random() - 1.0) * limit)
                weights.append(row)
            layers.append({
                "weights": weights,
                "biases": [0.0 for _ in range(fan_out)],
                "activation": output_activation if k == len(dims) - 2 else hidden_activation,
            })
        return FeedForwardNetwork(layers)

    def forward(self, x):
        activations = [list(x)]
        zs = []
        a = list(x)
        for layer in self.layers:
            z_layer = []
            a_layer = []
            for i, row in enumerate(layer["weights"]):
                z = layer["biases"][i]
                for j, w in enumerate(row):
                    z += w * a[j]
                z_layer.append(z)
                a_layer.append(activate(layer["activation"], z))
            zs.append(z_layer)
            activations.append(a_layer)
            a = a_layer
        return zs, activations

    def predict(self, x):
        return list(self.forward(x)[1][-1])

    def train_sample(self, x, y, lr):
        zs, activations = self.forward(x)
        pred = list(activations[-1])
        loss = 0.0
        d_a = []
        for i in range(len(pred)):
            err = pred[i] - y[i]
            loss += 0.5 * err * err
            d_a.append(err)

        for k in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[k]
            prev_a = activations[k]
            cur_a = activations[k + 1]
            cur_z = zs[k]
            delta = []
            for i, a in enumerate(cur_a):
                delta.append(d_a[i] * activation_prime(layer["activation"], a, cur_z[i]))

            d_prev = [0.0 for _ in prev_a]
            for i, row in enumerate(layer["weights"]):
                for j, w in enumerate(row):
                    d_prev[j] += w * delta[i]

            for i, row in enumerate(layer["weights"]):
                for j in range(len(row)):
                    row[j] -= lr * delta[i] * prev_a[j]
                layer["biases"][i] -= lr * delta[i]
            d_a = d_prev

        return loss, pred


def run_xor(seed, epochs, lr):
    net = FeedForwardNetwork.random(
        input_dim=2,
        hidden_layers=[4],
        output_dim=1,
        hidden_activation="tanh",
        output_activation="sigmoid",
        rng=Mulberry32(seed),
    )
    losses = []
    for _epoch in range(epochs):
        for x, y in XOR:
            loss, _pred = net.train_sample(x, y, lr)
            losses.append(loss)
    preds = [net.predict(x)[0] for x, _ in XOR]
    return {
        "config": {"seed": seed, "epochs": epochs, "learningRate": lr, "hiddenLayers": [4]},
        "predictions": preds,
        "lossHistory": losses,
        "avgLossLast100": sum(losses[-100:]) / 100.0,
    }


class Corridor:
    def __init__(self, length):
        self.num_states = length
        self.num_actions = 2
        self.goal = length - 1

    def step(self, state, action):
        if state == self.goal:
            return state, 0.0, True
        ns = state - 1 if action == 0 else state + 1
        ns = max(0, min(self.num_states - 1, ns))
        if ns == self.goal:
            return ns, 10.0, True
        return ns, -1.0, False


def corridor_optimal(length, gamma):
    env = Corridor(length)
    v = [0.0 for _ in range(length)]
    pi = [0 for _ in range(length)]
    for _ in range(5000):
        max_delta = 0.0
        for s in range(length):
            if s == env.goal:
                continue
            best_q = -1e100
            best_a = 0
            for a in range(env.num_actions):
                ns, reward, done = env.step(s, a)
                q = reward + (0.0 if done else gamma * v[ns])
                if q > best_q:
                    best_q = q
                    best_a = a
            max_delta = max(max_delta, abs(best_q - v[s]))
            v[s] = best_q
            pi[s] = best_a
        if max_delta < 1e-12:
            break
    return {"length": length, "gamma": gamma, "V": v, "policy": pi}


def rk4_decay(rate, y0, t1, dt):
    t = 0.0
    y = y0
    ts = [t]
    ys = [y]

    def f(_t, yy):
        return -rate * yy

    while t + 0.5 * dt < t1:
        k1 = f(t, y)
        k2 = f(t + dt / 2.0, y + dt * k1 / 2.0)
        k3 = f(t + dt / 2.0, y + dt * k2 / 2.0)
        k4 = f(t + dt, y + dt * k3)
        y = y + dt * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
        t = t + dt
        ts.append(t)
        ys.append(y)
    exact = y0 * math.exp(-rate * t1)
    return {
        "config": {"rate": rate, "y0": y0, "t1": t1, "dt": dt, "solver": "rk4"},
        "t": ts,
        "y": ys,
        "final": y,
        "exactFinal": exact,
        "error": abs(y - exact),
    }


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--xor-epochs", type=int, default=8000)
    parser.add_argument("--xor-lr", type=float, default=0.3)
    parser.add_argument("--corridor-length", type=int, default=6)
    parser.add_argument("--corridor-gamma", type=float, default=0.95)
    parser.add_argument("--ode-rate", type=float, default=0.5)
    parser.add_argument("--ode-y0", type=float, default=1.0)
    parser.add_argument("--ode-t1", type=float, default=2.0)
    parser.add_argument("--ode-dt", type=float, default=0.05)
    args = parser.parse_args(argv)

    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "tool": "dependency-free python reference",
        "xor": run_xor(args.seed, args.xor_epochs, args.xor_lr),
        "corridor": corridor_optimal(args.corridor_length, args.corridor_gamma),
        "neuralOdeDecay": rk4_decay(args.ode_rate, args.ode_y0, args.ode_t1, args.ode_dt),
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print("[neural-network] wrote " + str(out_path))
    print("[neural-network] XOR predictions: " + json.dumps([round(x, 4) for x in payload["xor"]["predictions"]]))
    print("[neural-network] corridor optimal policy: " + json.dumps(payload["corridor"]["policy"]))
    print("[neural-network] ODE error: %.3e" % payload["neuralOdeDecay"]["error"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
