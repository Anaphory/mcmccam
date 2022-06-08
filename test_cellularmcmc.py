from collections import defaultdict
import numpy
import pytest
from scipy import stats
from statsmodels.stats.rates import test_poisson_2indep as poisson_2indep

from cellularmcmc import History, HistoryModel, step_mcmc
from cellulartopology import make_block


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


@pytest.fixture(params=["line", "block", "star", "full"])
def ratematrix5(history5, request):
    n = len(history5.start())
    if request.param == "star":
        return numpy.vstack((
            numpy.hstack(([0], numpy.ones(n-1))),
            numpy.hstack((numpy.ones((n-1, 1)), numpy.zeros((n-1, n-1))))))
    elif request.param == "block":
        i = int(n**0.5)
        j = n - i*i
        return numpy.vstack((
            numpy.hstack((make_block(i, i), numpy.ones((i*i, j)))),
            numpy.ones((j, n))))
    elif request.param == "line":
        return numpy.diag(numpy.ones(n-1), 1) + numpy.diag(numpy.ones(n-1), -1)
    else:
        return numpy.ones((n, n)) - numpy.diag(numpy.ones(n))


@pytest.fixture(params=[1.0, 0.5, 1e-16])
def mu5(request):
    return request.param


@pytest.fixture(params=[3, 6, 200])
def nlang5(request):
    return request.param


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
                    logp_cond[potential_node, alternative] = m.calculate_change_loglikelihood(
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
        assert numpy.allclose(numpy.nan_to_num(p_raw / numpy.exp(logp_cond), q, q, q), q)


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
        alternatives = list(m.alternatives_with_likelihood(history5, time, node))
        # An alternative can be excluded because it is the previous or the next
        # state of the node; if the previous and next state are identical,
        # there is only one forbidden state.
        assert nlang5 - 2 <= len(alternatives) <= nlang5 - 1
        ps = []
        likelihoods = []
        for alternative, p in alternatives:
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
                p0 = p

            likelihoods.append(m.loglikelihood(alternative_history))
            ps.append(p)
        assert numpy.allclose(
            numpy.asarray(ps) / p0, numpy.exp(numpy.asarray(likelihoods) - reference)
        )


class MCMCTest:
    def __init__(self):
        self.stats = defaultdict(list)
        self.true = 32
        self.mcmc = 64
        self.mcmc_steps = 200
        self.significance = 1e-4

    def stats(self, history: History):
        yield "n_changes", len(history.changes)

    def compute_statistics(self, history):
        yield "n_changes", len(history.all_changes())

    def gather_statistics(self, ground_truth: bool, history: History):
        for key, value in self.compute_statistics(history):
            self.stats[ground_truth, key].append(value)

    def test_statistics(self):
        tests = {}
        # C test for Poisson means
        true_changes = self.stats[True, "n_changes"]
        test_changes = self.stats[False, "n_changes"]
        tests["n_changes are the same"] = (
            poisson_2indep(
                sum(true_changes), self.true, sum(test_changes), self.mcmc
            ).pvalue
            >= self.significance
        )
        return tests

    def __call__(self, ratematrix5, mu5, nlang5):
        m = HistoryModel(ratematrix5, mu5, range(nlang5))
        for i in range(self.true):
            self.gather_statistics(True, m.generate_history([0 for _ in range(5)], 10))
        for i in range(self.mcmc):
            h = m.generate_history([0 for _ in range(5)], 10)
            lk = m.loglikelihood(h)
            for i in range(self.mcmc_steps):
                h, lk = step_mcmc(m, h, lk)
            self.gather_statistics(False, h)

        assert all(self.test_statistics().values())


def test_mcmc(ratematrix5, mu5, nlang5):
    test = MCMCTest()
    test(ratematrix5, mu5, nlang5)
