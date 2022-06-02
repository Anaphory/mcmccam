import numpy
import pytest

from cellularmcmc import History


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


@pytest.fixture()
def ratematrix5():
    return numpy.ones((5, 5)) - numpy.diag(numpy.ones(5))


@pytest.fixture(params=[0.5])
def mu5(request):
    return request.param


@pytest.fixture(params=[6, 3])
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
    for time, node, value in history5.all_changes():
        p_cond = numpy.zeros((len(history5.start()), nlang5))
        p_raw = numpy.zeros((len(history5.start()), nlang5))
        for potential_node in range(len(history5.start())):
            for alternative in range(nlang5):
                p_raw[
                    potential_node, alternative
                ] = history5.calculate_event_likelihood(
                    time, potential_node, alternative, ratematrix5, mu5, nlang5
                )
                try:
                    p_cond[
                        potential_node, alternative
                    ] = history5.calculate_change_likelihood(
                        time, potential_node, alternative, ratematrix5, mu5, nlang5
                    )
                except ValueError:
                    continue
        q = 1 - history5.calculate_no_change_likelihood(time, ratematrix5, mu5, nlang5)
        assert numpy.allclose(p_raw.sum(), 1)
        assert numpy.allclose(p_cond.sum(), 1)
        assert numpy.allclose(numpy.nan_to_num(p_raw / p_cond, q, q, q), q)


def test_alternative_with_likelihood(history5, ratematrix5, mu5, nlang5):
    """
    The probabilites in alternatives_with_likelihood are calculated such that

        exp(h.loglikelihood(...)) / alternatives_with_likelihood(h, node, time ...)

    is constant for all changes (node, time).
    """
    history = history5
    copy_rate_matrix = ratematrix5
    mutation_rate = mu5
    nlang = nlang5

    reference = history.loglikelihood(copy_rate_matrix, mutation_rate, nlang)
    changes = history.all_changes()
    for i, (time, node, value) in enumerate(changes):
        alternatives = list(
            history.alternatives_with_likelihood(
                node, time, copy_rate_matrix, mutation_rate, nlang
            )
        )
        # An alternative can be excluded because it is the previous or the next
        # state of the node; if the previous and next state are identical,
        # there is only one forbidden state.
        assert nlang - 2 <= len(alternatives) <= nlang - 1
        ps = []
        likelihoods = []
        for alternative, p in alternatives:
            try:
                alternative_history = History(
                    history.start(),
                    [
                        (time, node, value if j != i else alternative)
                        for j, (time, node, value) in enumerate(changes)
                    ],
                    history.end,
                )
            except ValueError:
                continue

            if alternative == value:
                p0 = p

            likelihoods.append(
                alternative_history.loglikelihood(
                    copy_rate_matrix, mutation_rate, nlang
                )
            )
            ps.append(p)
        assert numpy.allclose(
            numpy.asarray(ps) / p0, numpy.exp(numpy.asarray(likelihoods) - reference)
        )

