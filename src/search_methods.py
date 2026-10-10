"""
search_methods.py
The four wrapper search methods and the budget-tracking evaluator they all share.

Key rule: a search method only ever sees the Evaluator. The Evaluator only holds
inner folds built from the TRAINING part of the outer split, so no search method
can ever touch the held-out test rows.

Budget rule: every call to evaluator.score() costs one evaluation, even if the
same subset was scored before (this is the usual "function evaluations" count in
the metaheuristic literature). Repeated subsets are looked up in a cache only to
save time; they still use up budget, so no method gets free evaluations.
"""

import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score

from utils import make_classifier


class Evaluator:
    """Scores feature subsets with inner cross-validation and records the
    best-so-far subset every time the evaluation count reaches a checkpoint."""

    def __init__(self, scaled_folds, n_features, cfg):
        self.parts = scaled_folds.parts
        self.n_features = n_features
        self.alpha = cfg["alpha"]
        self.cfg = cfg
        self.max_evals = cfg["max_evaluations"]
        self.checkpoints = list(cfg["checkpoints"])

        self.count = 0
        self.cache = {}
        self.best_mask = None
        self.best_fitness = np.inf
        self.best_inner_ba = np.nan
        self.snapshots = []                   # one dict per checkpoint
        self.start = time.perf_counter()

    @property
    def done(self):
        return self.count >= self.max_evals

    def _inner_ba(self, mask):
        cols = np.flatnonzero(mask)
        scores = []
        for X_tr, y_tr, X_va, y_va in self.parts:
            model = make_classifier(self.cfg, len(y_tr)).fit(X_tr[:, cols], y_tr)
            scores.append(balanced_accuracy_score(y_va, model.predict(X_va[:, cols])))
        return float(np.mean(scores))

    def score(self, mask):
        """Return the fitness of a subset (lower is better). Costs one evaluation."""
        if self.done:
            raise RuntimeError("budget already used up; search methods must check .done")
        mask = np.asarray(mask, dtype=bool)
        if not mask.any():
            raise ValueError("empty subset; repair it before scoring")

        key = mask.tobytes()
        if key not in self.cache:
            ba = self._inner_ba(mask)
            fit = self.alpha * (1.0 - ba) + (1.0 - self.alpha) * mask.sum() / self.n_features
            self.cache[key] = (fit, ba)
        fit, ba = self.cache[key]
        self.count += 1

        # Strictly better fitness wins; on an exact tie the smaller subset wins.
        better = fit < self.best_fitness or (
            fit == self.best_fitness and mask.sum() < self.best_mask.sum()
        )
        if better:
            self.best_fitness, self.best_inner_ba, self.best_mask = fit, ba, mask.copy()

        if len(self.snapshots) < len(self.checkpoints) and self.count == self.checkpoints[len(self.snapshots)]:
            self._snapshot(self.count)
        return fit

    def _snapshot(self, budget):
        self.snapshots.append({
            "checkpoint": budget,
            "evals_used": self.count,
            "mask": self.best_mask.copy(),
            "inner_ba": self.best_inner_ba,
            "fitness": self.best_fitness,
            "seconds": time.perf_counter() - self.start,
        })

    def finish(self):
        """If a method stopped early (SFS can), the remaining checkpoints simply
        keep the final best subset; evals_used shows how much was really spent."""
        while len(self.snapshots) < len(self.checkpoints):
            self._snapshot(self.checkpoints[len(self.snapshots)])
        return self.snapshots


# ---------------- helpers ----------------

def repair(mask, rng):
    """An empty subset is not allowed; switch on one random feature."""
    if not mask.any():
        mask[rng.integers(mask.size)] = True
    return mask


def random_population(size, n_features, rng):
    pop = rng.random((size, n_features)) < 0.5
    for row in pop:
        repair(row, rng)
    return pop


# ---------------- 1. Genetic algorithm ----------------

def genetic_algorithm(ev, rng, cfg):
    p = cfg["ga"]
    size, D = cfg["population_size"], ev.n_features
    pm = 1.0 / D                                      # mutation rate per bit

    pop = random_population(size, D, rng)
    fit = np.full(size, np.inf)
    for i in range(size):
        if ev.done:
            return
        fit[i] = ev.score(pop[i])

    def tournament():
        picks = rng.choice(size, size=p["tournament_size"], replace=False)
        return pop[picks[np.argmin(fit[picks])]]

    while not ev.done:
        order = np.argsort(fit, kind="stable")
        new_pop = [pop[i].copy() for i in order[:p["elite"]]]
        new_fit = [fit[i] for i in order[:p["elite"]]]   # elites keep their score (no re-evaluation)

        while len(new_pop) < size and not ev.done:
            a, b = tournament(), tournament()
            if rng.random() < p["crossover_rate"]:
                child = np.where(rng.random(D) < 0.5, a, b)       # uniform crossover
            else:
                child = a.copy()
            flip = rng.random(D) < pm
            child = repair(child ^ flip, rng)
            new_fit.append(ev.score(child))
            new_pop.append(child)

        if len(new_pop) < size:          # budget ran out mid-generation
            return
        pop, fit = np.array(new_pop), np.array(new_fit)


# ---------------- 2. Binary particle swarm (Kennedy & Eberhart, 1997) ----------------

def binary_pso(ev, rng, cfg):
    p = cfg["pso"]
    size, D = cfg["population_size"], ev.n_features

    x = random_population(size, D, rng)
    v = np.zeros((size, D))
    pbest, pbest_fit = x.copy(), np.full(size, np.inf)
    for i in range(size):
        if ev.done:
            return
        pbest_fit[i] = ev.score(x[i])
    g = int(np.argmin(pbest_fit))

    while not ev.done:
        for i in range(size):            # particles move one at a time, so the budget stops cleanly
            if ev.done:
                return
            progress = ev.count / ev.max_evals
            w = p["w_start"] - (p["w_start"] - p["w_end"]) * progress
            r1, r2 = rng.random(D), rng.random(D)
            v[i] = (w * v[i]
                    + p["c1"] * r1 * (pbest[i].astype(float) - x[i])
                    + p["c2"] * r2 * (pbest[g].astype(float) - x[i]))
            v[i] = np.clip(v[i], -p["v_max"], p["v_max"])
            x[i] = repair(rng.random(D) < 1.0 / (1.0 + np.exp(-v[i])), rng)
            f = ev.score(x[i])
            if f < pbest_fit[i]:
                pbest[i], pbest_fit[i] = x[i].copy(), f
                if f < pbest_fit[g]:
                    g = i


# ---------------- 3. Random search ----------------

def random_search(ev, rng, cfg):
    """Each try picks a subset size uniformly from 1..D, then that many random features.
    (Switching each feature on with probability 0.5 would almost always give subsets
    of about D/2, which would be an unfairly narrow random search.)"""
    D = ev.n_features
    while not ev.done:
        mask = np.zeros(D, dtype=bool)
        mask[rng.choice(D, size=rng.integers(1, D + 1), replace=False)] = True
        ev.score(mask)


def random_search_bernoulli(ev, rng, cfg):
    """Revision check: each feature is switched on with probability 0.5, exactly like the
    starting subsets of GA and BPSO. An empty draw is redrawn (it cannot be scored)."""
    D = ev.n_features
    while not ev.done:
        mask = rng.random(D) < 0.5
        if mask.any():
            ev.score(mask)


# ---------------- 4. Sequential forward selection ----------------

def sequential_forward(ev, rng, cfg):
    """Start empty; at each step try adding every unused feature and keep the best.
    Stop when no addition improves the fitness, or when the budget runs out.
    Candidates are tried in a random order so that, if the budget ends mid-step,
    no feature is favoured just because of its column number."""
    D = ev.n_features
    current = np.zeros(D, dtype=bool)
    current_fit = np.inf
    while not ev.done and not current.all():
        best_f, best_j = np.inf, None
        for j in rng.permutation(np.flatnonzero(~current)):
            if ev.done:
                break
            trial = current.copy()
            trial[j] = True
            f = ev.score(trial)
            if f < best_f:
                best_f, best_j = f, j
        if best_j is None or best_f >= current_fit:
            return
        current[best_j], current_fit = True, best_f


SEARCH_METHODS = {
    "ga": genetic_algorithm,
    "pso": binary_pso,
    "random": random_search,
    "random_bern": random_search_bernoulli,     # revision check only
    "sfs": sequential_forward,
}