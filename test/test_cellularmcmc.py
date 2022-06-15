import json
import tempfile
from collections import defaultdict

import numpy
import pytest
from cellulartopology import make_block
from scipy import stats
from statsmodels.stats.rates import test_poisson_2indep as poisson_2indep

from cellularmcmc import History
from cellularmcmc import HistoryModel
from cellularmcmc import step_mcmc


@pytest.fixture(
    params=[
        (numpy.zeros(5, dtype=int), [], 1.0),
        (numpy.zeros(5, dtype=int), [(0.5, 0, 1)], 1.0),
        (numpy.zeros(5, dtype=int), [(0.5, 0, 1), (0.6, 1, 2)], 1.0),
        (numpy.zeros(5, dtype=int), [(0.1, 0, 1), (0.2, 1, 1), (0.3, 2, 1)], 1.0),
        (
            numpy.zeros(5, dtype=int),
            [(0.1, 0, 1), (0.2, 1, 1), (0.3, 2, 1), (0.4, 0, 0)],
            1.0,
        ),
    ]
)
def history5(request):
    return History(*request.param)


@pytest.fixture(params=["line", "block", "star", "shift", "full"])
def ratematrix5(history5, request):
    n = len(history5.start())
    if request.param == "star":
        return numpy.vstack(
            (
                numpy.hstack(([0], numpy.ones(n - 1))),
                numpy.hstack((numpy.ones((n - 1, 1)), numpy.zeros((n - 1, n - 1)))),
            )
        )
    elif request.param == "block":
        i = int(n**0.5)
        j = n - i * i
        return numpy.vstack(
            (
                numpy.hstack((make_block(i, i), numpy.ones((i * i, j)))),
                numpy.ones((j, n)),
            )
        )
    elif request.param == "line":
        return numpy.diag(numpy.ones(n - 1), 1) + numpy.diag(numpy.ones(n - 1), -1)
    elif request.param == "shift":
        array = numpy.diag(numpy.ones(n - 1), 1)
        array[-1, 0] = 1
        assert (array.sum(1) == 1).all()
        return array
    else:
        return numpy.ones((n, n)) - numpy.diag(numpy.ones(n))


@pytest.fixture(params=[1.0, 0.5, 1e-16])
def mu5(request):
    return request.param


@pytest.fixture(params=[3, 6, 200])
def nlang5(request):
    return request.param


def test_REG1():
    history = History([0, 0, 0, 0, 0], [(0.5, 0, 1)], 1.0)
    model = HistoryModel(
        copy_rate_matrix=numpy.array(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        ),
        mutation_rate=1e-16,
        languages=range(0, 3),
    )

    assert numpy.allclose(
        numpy.exp(model.calculate_event_loglikelihood(history, 0.4, 0, 0)), 1 / 8
    )


def test_p_any_change_leq_1(history5, ratematrix5, mu5, nlang5):
    m = HistoryModel(ratematrix5, mu5, range(nlang5))
    for time, node, value in history5.all_changes():
        assert m.calculate_any_change_loglikelihood(history5, time) <= 0
        assert m.calculate_no_change_likelihood(history5, time) <= 1
        assert numpy.allclose(
            numpy.exp(m.calculate_any_change_loglikelihood(history5, time))
            + m.calculate_no_change_likelihood(history5, time),
            1.0,
        )


def test_likelihoods_add_up(history5, ratematrix5, mu5, nlang5):
    """Test that local likelihoods add up.

    Test that the conditional probabilities of events, given that an event
    happens at a specific time, add up to 1, and that the probabilites of changes,
    given that a change happens at a specific time, also add up to 1.

    Check that these two probabilities are proportional to each other, and that
    the proportionality factor is exactly given by the likelihood of seeing a
    change given an event.

    """
    m = HistoryModel(ratematrix5, mu5, range(nlang5))
    for time, node, value in history5.all_changes():
        logp_cond = numpy.full((len(history5.start()), nlang5), -numpy.inf)
        p_raw = numpy.zeros((len(history5.start()), nlang5))
        for potential_node in range(len(history5.start())):
            for alternative in range(nlang5):
                p_raw[potential_node, alternative] = m.calculate_event_likelihood(
                    history5,
                    time,
                    potential_node,
                    alternative,
                )
                try:
                    logp_cond[
                        potential_node, alternative
                    ] = m.calculate_change_loglikelihood(
                        history5,
                        time,
                        potential_node,
                        alternative,
                    )
                except ValueError:
                    continue
        q = 1 - m.calculate_no_change_likelihood(history5, time)
        assert numpy.allclose(p_raw.sum(), 1)
        assert numpy.allclose(numpy.exp(logp_cond).sum(), 1)
        assert numpy.allclose(
            numpy.nan_to_num(p_raw / numpy.exp(logp_cond), q, q, q), q
        )


def test_alternative_with_likelihood(history5, ratematrix5, mu5, nlang5):
    """
    The probabilites in alternatives_with_likelihood are calculated such that

        exp(h.loglikelihood(...)) / alternatives_with_likelihood(h, node, time ...)

    is constant for all changes (node, time).
    """
    m = HistoryModel(ratematrix5, mu5, range(nlang5))
    reference = m.loglikelihood(history5)
    changes = history5.all_changes()
    for i, (time, node, value) in enumerate(changes):
        alternatives = list(m.alternatives_with_loglikelihood(history5, time, node))
        # An alternative can be excluded because it is the previous or the next
        # state of the node; if the previous and next state are identical,
        # there is only one forbidden state.
        assert nlang5 - 2 <= len(alternatives) <= nlang5 - 1
        logps = []
        likelihoods = []
        for alternative, logp in alternatives:
            try:
                alternative_history = History(
                    history5.start(),
                    [
                        (time, node, value if j != i else alternative)
                        for j, (time, node, value) in enumerate(changes)
                    ],
                    history5.end,
                )
            except ValueError:
                continue

            if alternative == value:
                logp0 = logp

            likelihoods.append(m.loglikelihood(alternative_history))
            logps.append(logp)
        assert numpy.allclose(
            numpy.asarray(logps) - logp0, numpy.asarray(likelihoods) - reference
        )


class MCMCTest:
    def __init__(self):
        self.true_stats = defaultdict(list)
        self.mcmc_stats = defaultdict(list)
        self.true = 32
        self.mcmc = 64
        self.mcmc_steps = 200
        self.significance = 1e-4

    def compute_statistics(self, history):
        yield "n_changes", len(history.all_changes())
        try:
            yield "time_until_first_change", history.all_changes()[0][0]
        except IndexError:
            yield "time_until_first_change", numpy.inf
        yield "n_changes_node0", len([1 for t, n, v in history.all_changes() if n == 0])

    def gather_statistics(self, ground_truth: bool, history: History):
        for key, value in self.compute_statistics(history):
            if ground_truth:
                self.true_stats[key].append(value)
            else:
                self.mcmc_stats[key].append(value)

    def test_statistics(self):
        tests = {}

        # C test for Poisson means
        true_changes = self.true_stats["n_changes"]
        test_changes = self.mcmc_stats["n_changes"]
        tests["n_changes are the same"] = (
            poisson_2indep(
                sum(true_changes), self.true, sum(test_changes), self.mcmc
            ).pvalue
            >= self.significance
        )

        # Perform Kolmorogorov-Smirnov-tests for all those stats.
        for key, value in self.true_stats.items():
            tests[f"Kolmogorov-Smirnov {key}"] = (
                stats.kstest(self.mcmc_stats[key], value).pvalue >= self.significance
            )

        return tests

    def __call__(self, ratematrix5, mu5, nlang5):
        m = HistoryModel(ratematrix5, mu5, range(nlang5))
        # t = tqdm(total=self.true + self.mcmc*self.mcmc_steps)
        for i in range(self.true):
            self.gather_statistics(True, m.generate_history([0 for _ in range(5)], 10))
            # t.update()
        for i in range(self.mcmc):
            h = m.generate_history([0 for _ in range(5)], 10)
            lk = m.loglikelihood(h)
            for i in range(self.mcmc_steps):
                h, lk = step_mcmc(m, h, lk)
                # t.update()
            self.gather_statistics(False, h)

        _, path = tempfile.mkstemp(".json", "report-")
        with open(path, "w") as statsfile:
            json.dump(
                {
                    key: {"True": value, "MCMC": self.mcmc_stats.get(key)}
                    for key, value in self.true_stats.items()
                },
                statsfile,
                indent=2,
                sort_keys=True,
            )
        assert all(
            self.test_statistics().values()
        ), f"Some stats don't match between MCMC and direct generation. Check {path} for details."


def test_mcmc(ratematrix5, mu5, nlang5, capsys):
    test = MCMCTest()
    with capsys.disabled():
        test(ratematrix5, mu5, nlang5)
