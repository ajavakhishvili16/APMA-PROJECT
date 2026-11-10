"""Random forest regression written from scratch, line by line from our pseudocode.

    python forest_from_scratch.py          (about 1-2 minutes)

The main analysis (random_forest.py) uses scikit-learn's RandomForestRegressor
because it is fast enough for 500 trees. This file implements the same algorithm
ourselves, so every step of the pseudocode on the slides is visible as code:

  Algorithm 1  RandomForest   bootstrap sample per tree, then average the trees
  Algorithm 2  grow_tree      stopping rule, draw m inputs, best split, recurse
  Algorithm 3  best_split     sort, prefix sums, gain for every cutoff in O(n)

Each block is labelled with the algorithm line it implements. At the bottom,
the script checks the result against scikit-learn:
  1. the best root split on the full training data (input, cutoff and gain)
     is the same as scikit-learn's to rounding error, and
  2. a 50-tree forest of ours scores about the same 2026 test R^2 as a
     50-tree scikit-learn forest with the same settings.
"""
import time

import numpy as np


# ---------------------------------------------------------------------------
# Algorithm 3: exact best split of one input j inside a node
# ---------------------------------------------------------------------------
def best_split(x, y, min_leaf):
    """Return (gain, cutoff) of the best split x <= s vs x > s, or (0, None).

    x, y: the node's values of input j and of the response, both length n.
    Gain = (n_L n_R / n) (ybar_L - ybar_R)^2, the drop in squared error (Part 3).
    """
    n = len(y)
    order = np.argsort(x, kind="stable")          # line 1: sort by x_j, O(n log n)
    xs, ys = x[order], y[order]
    S = np.cumsum(ys)                             # line 2: prefix sums S_k
    total = S[-1]
    k = np.arange(1, n)                           # candidate left sizes k = 1..n-1
    ok = (k >= min_leaf) & (n - k >= min_leaf) & (xs[k - 1] < xs[k])   # line 3: the set K
    if not ok.any():
        return 0.0, None
    k = k[ok]
    ybar_L = S[k - 1] / k                         # line 4: child means in O(1) each
    ybar_R = (total - S[k - 1]) / (n - k)
    gain = k * (n - k) / n * (ybar_L - ybar_R) ** 2
    best = np.argmax(gain)                        # line 6: k* = argmax gain
    kb = k[best]
    return float(gain[best]), float((xs[kb - 1] + xs[kb]) / 2)   # cutoff s* = midpoint


# ---------------------------------------------------------------------------
# Algorithm 2: grow one regression tree on the sales in idx (a multiset)
# ---------------------------------------------------------------------------
def grow_tree(X, y, idx, m, min_leaf, rng, tree=None):
    """Grow a tree recursively. Nodes are stored in a dict of lists; returns node id."""
    if tree is None:
        tree = {"feature": [], "cutoff": [], "left": [], "right": [], "value": [], "n": []}
    node = len(tree["value"])
    for key in tree:
        tree[key].append(-1)
    n = len(idx)
    yN = y[idx]
    tree["value"][node] = yN.mean()               # leaf value c_N = mean of y (Part 3)
    tree["n"][node] = n

    if n < 2 * min_leaf:                          # line 1: stopping rule
        return node
    J = rng.choice(X.shape[1], size=m, replace=False)   # line 2: draw m of the p inputs (Part 2)

    best_gain, best_j, best_s = 0.0, None, None
    for j in J:                                   # line 4: argmax over j in J and cutoffs s
        g, s = best_split(X[idx, j], yN, min_leaf)    # (Algorithm 3)
        if s is not None and g > best_gain:
            best_gain, best_j, best_s = g, j, s
    if best_j is None:                            # line 5: no feasible split -> leaf
        return node

    go_left = X[idx, best_j] <= best_s            # line 3: N_L = {x_ij <= s}, N_R = {x_ij > s}
    tree["feature"][node], tree["cutoff"][node] = int(best_j), best_s
    tree["left"][node] = grow_tree(X, y, idx[go_left], m, min_leaf, rng, tree)    # line 6: recurse
    tree["right"][node] = grow_tree(X, y, idx[~go_left], m, min_leaf, rng, tree)
    return node


def predict_tree(tree, X):
    """T_b(x): send every row down the tree; return the value of the leaf it reaches."""
    out = np.empty(len(X))
    stack = [(0, np.arange(len(X)))]
    while stack:
        node, rows = stack.pop()
        if tree["feature"][node] == -1:           # leaf
            out[rows] = tree["value"][node]
            continue
        left = X[rows, tree["feature"][node]] <= tree["cutoff"][node]
        stack.append((tree["left"][node], rows[left]))
        stack.append((tree["right"][node], rows[~left]))
    return out


# ---------------------------------------------------------------------------
# Algorithm 1: the forest
# ---------------------------------------------------------------------------
class RandomForest:
    def __init__(self, B=500, m=9, min_leaf=5, seed=0):
        self.B, self.m, self.min_leaf = B, m, min_leaf
        self.rng = np.random.default_rng(seed)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n = len(y)
        self.trees = []
        for b in range(self.B):                   # line 1: for b = 1..B
            I_b = self.rng.integers(0, n, size=n) # line 2: n draws, iid Unif{1..n} (Part 1)
            tree = {"feature": [], "cutoff": [], "left": [], "right": [], "value": [], "n": []}
            grow_tree(X, y, I_b, self.m, self.min_leaf, self.rng, tree)   # line 3: T_b = GrowTree(I_b)
            self.trees.append(tree)
        return self

    def tree_predictions(self, X):
        X = np.asarray(X, float)
        return np.array([predict_tree(t, X) for t in self.trees])

    def predict(self, X):
        return self.tree_predictions(X).mean(axis=0)   # line 5: f_hat(x) = (1/B) sum_b T_b(x) (Part 4)


# ---------------------------------------------------------------------------
# Check against scikit-learn on our data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor

    from common import BASE, TEST, TRAIN, load, metrics
    from random_forest import forest_inputs

    train, test = load(TRAIN), load(TEST)
    levels = {c: sorted(train[c].astype(str).unique()) for c in BASE}
    Xtr_df = forest_inputs(train, levels)
    names = list(Xtr_df.columns)
    Xtr, Xte = Xtr_df.values, forest_inputs(test, levels).values
    ytr, yte = train.y.values, test.y.values
    n, p = Xtr.shape
    m = int(0.33 * p)                              # 9 of 28, as in random_forest.py
    print(f"training sales n = {n}, inputs p = {p}, m = {m}, min leaf = 5")

    # 1. Best root split on all training sales, all inputs (Algorithm 3 for every j)
    gains = [best_split(Xtr[:, j], ytr, 5) for j in range(p)]
    j_star = int(np.argmax([g for g, _ in gains]))
    g_ours, s_ours = gains[j_star]
    sk = DecisionTreeRegressor(max_depth=1, min_samples_leaf=5).fit(Xtr, ytr)
    t = sk.tree_
    nL, nR = t.n_node_samples[1], t.n_node_samples[2]
    g_sk = n * t.impurity[0] - nL * t.impurity[1] - nR * t.impurity[2]   # SSE_parent - SSE_L - SSE_R
    print("\n1. Best root split (all sales, all inputs)")
    print(f"   ours:         {names[j_star]} <= {s_ours:.4f}, gain {g_ours:.4f}")
    print(f"   scikit-learn: {names[t.feature[0]]} <= {t.threshold[0]:.4f}, gain {g_sk:.4f}")
    print(f"   difference in gain: {abs(g_ours - g_sk):.2e}")

    # 2. Small forests, same settings, scored on the 2026 test set
    B = 50
    t0 = time.time()
    ours = RandomForest(B=B, m=m, min_leaf=5, seed=0).fit(Xtr, ytr)
    secs = time.time() - t0
    P = ours.tree_predictions(Xte)
    running = np.cumsum(P, axis=0) / np.arange(1, B + 1)[:, None]
    skf = RandomForestRegressor(B, max_features=0.33, min_samples_leaf=5, n_jobs=-1,
                                random_state=0).fit(Xtr, ytr)
    print(f"\n2. {B}-tree forests on the 2026 test set (ours took {secs:.0f} s to grow)")
    for b in [1, 10, B]:
        print(f"   ours, B = {b:>2}: R^2 {metrics(yte, running[b - 1])['R2']:.3f}")
    print(f"   scikit-learn, B = {B}: R^2 {metrics(yte, skf.predict(Xte))['R2']:.3f}")
    depth = []
    for tr in ours.trees:   # depth of each leaf, via parent links
        d = {0: 0}
        for i, (l, r) in enumerate(zip(tr["left"], tr["right"])):
            if l != -1:
                d[l] = d[r] = d[i] + 1
        depth.append(max(d.values()))
    leaves = [sum(f == -1 for f in tr["feature"]) for tr in ours.trees]
    print(f"   our trees: median depth {np.median(depth):.0f}, median leaves {np.median(leaves):.0f}")
    print("\nThe two forests differ only in their random draws (and scikit-learn counts the")
    print("minimum leaf size in distinct sales rather than draws), so R^2 agrees closely, not exactly.")
