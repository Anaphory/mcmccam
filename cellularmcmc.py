import numpy
import deepcopy
import typing
import itertools
import bisect
from matplotlib import pyplot as plt
from scipy.stats import expon

L = typing.TypeVar("L", bound=typing.Hashable)


class History(typing.Generic[L]):
    def __init__(
        self, start: typing.Sequence[L], changes: typing.Iterable[tuple[float, int, L]]
    ) -> None:
        """Store a history.

        A history represents a starting state at time t=0.0, and a set of
        changes (t, n, v), where at time t the node at index n takes the new
        value v.

        The times are expected to be sorted; there is undefined behaviour when
        they are not.

        """
        start = numpy.asarray(start)
        self._times = numpy.full((1, len(start)), numpy.inf)
        self._states = numpy.zeros((0, len(start)), dtype=start.dtype)
        history_depth = {}
        for time, node, new_value in changes:
            try:
                history_depth[node] += 1
            except KeyError:
                history_depth[node] = 0
            if history_depth[node] + 1 >= self._times.shape[0]:
                self._times = numpy.vstack(
                    (self._times, numpy.full((1, len(start)), numpy.inf))
                )
                self._states = numpy.vstack(
                    (self._states, numpy.zeros((1, len(start)), dtype=start.dtype))
                )
            self._times[history_depth[node], node] = time
            self._states[history_depth[node], node] = new_value
        self._states = numpy.vstack((start[None, :], self._states))

    def all_changes(self) -> typing.Iterable[tuple[float, int, L]]:
        """List all changes documented in this history.

        Extract the times, nodes, and values of changes, similar to the
        iterable given to the constructor.

        >>> random_times = numpy.cumsum(numpy.random.random(10))
        >>> random_nodes = numpy.random.randint(3, size=10)
        >>> random_values = numpy.random.randint(30, size=10)
        >>> h = History(
        ...   numpy.array([0, 0, 0], int),
        ...   zip(random_times, random_nodes, random_values))
        >>> all(random_times == [t for t, _, _ in h.all_changes()])
        True
        >>> all(random_nodes == [n for _, n, _ in h.all_changes()])
        True
        >>> all(random_values == [v for _, _, v in h.all_changes()])
        True

        """
        return sorted(
            (time, node := numpy.unravel_index(nodeslice, self._times.shape)[1], state)
            for nodeslice, (time, state) in enumerate(
                zip(self._times[:-1].flat, self._states[1:].flat)
            )
            if time < numpy.inf
        )

    def at(self, node: int, time: float) -> L:
        """Look up the value of a node at a particular time.

        Examples
        ========

        >>> h = History(
        ...   numpy.array(["A", "A"], dtype="<U1"),
        ...   [(0.1, 0, "C"), (0.25, 1, "C"), (0.4, 0, "B")])

        So at the start, every node has value A.

        >>> h.at(0, 0.0)
        'A'
        >>> h.at(1, 0.0)
        'A'

        At time 0.3, both nodes have changed to C.

        >>> h.at(0, 0.3)
        'C'
        >>> h.at(1, 0.3)
        'C'

        For the exact time of a change, the function reports the new value.

        >>> h.at(1, 0.25)
        'C'


        In the end, the nodes have changed to B and C.

        >>> h.at(0, 999.)
        'B'
        >>> h.at(1, 999.)
        'C'

        Going back before the start is also possible, and just returns the
        starting values.

        >>> h.at(0, -999.)
        'A'
        >>> h.at(1, -999.)
        'A'
        """
        time_slice = bisect.bisect(self._times[:, node], time)
        return self._states[time_slice, node]

    def before(self, node: int, time: float) -> L:
        """Look up the value of a node infinitesimally before a particular time.

        In most cases, this is the same as History.at(), except when the node
        changes in the given instance. In that case, History.before() gives the
        old value, while History.at() gives the new value.

        Examples
        ========

        >>> h = History(
        ...   numpy.array(["A", "A"], dtype="<U1"),
        ...   [(0.1, 0, "C"), (0.25, 1, "C"), (0.4, 0, "B")])

        >>> for node, time in [
        ...     (0, 0.0), (1, 0.0), (0, 0.3), (1, 0.3),
        ...     (0, 999.), (1, 999.), (0, -999.), (1, -999.)]:
        ...   assert h.at(node, time) == h.before(node, time)

        For the exact time of a change, the function reports the old value.

        >>> h.before(0, 0.1)
        'A'
        >>> h.before(1, 0.25)
        'A'
        >>> h.before(0, 0.4)
        'C'

        """
        time_slice = bisect.bisect_left(self._times[:, node], time)
        return self._states[time_slice, node]

    def calculate_event_likelihood(
        self,
        node: int,
        new_state: L,
        time: float,
        copy_rate_matrix: numpy.ndarray,
        mutation_rate: float,
        nlang: int,
    ):
        """Calculate the likelihood of an event at some point in the history.

        This only calculates the likelihood that at a specific time point, a
        particular node takes the given new value based on earlier events. It
        does not compute the marginal likelihood of that change based on later
        changes. The likelihood is not conditioned on the event being
        observable as an actual change in the system state.

        See also
        ========

        History.calculate_change_likelihood:
            The same likelihood calculation, conditioned on the fact than an
            event is an actual change.

        Examples
        ========

        1: Copying, without mutation
        ~~~~~~~~~~~~~~~~~~~~~~~~~~~~

        Imagine three equally connected nodes
        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        that start in three different states
        >>> start = ["A", "B", "C"]

        out of a total of 5 different states
        >>> nlang = 5

        and no history of change yet.

        Assume no mutations, and also assume that no changes have happened yet.

        >>> h = History(start, [])

        If the first change is at time 0.2, it will have equal probability of
        changing node 0 to B (copied from node 1) or C (copied from node 2);
        changing node 1 to A or C, or changing node 2 to A or B – so each has
        probability 1/6.

        >>> h.calculate_event_likelihood(0, "B", 0.2, matrix, 0.0, 5)
        0.16666666666666666
        >>> h.calculate_event_likelihood(1, "A", 0.2, matrix, 0.0, 5)
        0.16666666666666666

        No other change has positive probability.
        >>> for i in "A", "D", "E":
        ...   print(h.calculate_event_likelihood(0, "D", 0.2, matrix, 0.0, 5))
        0.0
        0.0
        0.0

        2: Copying or mutation
        ~~~~~~~~~~~~~~~~~~~~~~

        Assume the same example as above, but now assume that mutations happen
        with half the rate of anything being copied.

        >>> mu = 0.5

        Then each of the three nodes still has the same probability of being
        the target of a change (1/3), but each copying is twice as likely as a
        random mutation, so if we focus on node 0 as the target of a change: In
        2 out of 5 cases it copies a B from node 1, in 2/5 cases it copies a C
        from node 2, and with probability 1/25 each, it takes a new value of A,
        B, C, D, or E.

        So the probability of a change to A, D, or E is 1/(25*3), whereas a
        change to B or C happens in 2/(5*3) + 1/(25*3) of cases.

        >>> h.calculate_event_likelihood(0, "B", 0.2, matrix, mu, 5)
        0.14666666666666667
        >>> h.calculate_event_likelihood(0, "C", 0.2, matrix, mu, 5)
        0.14666666666666667
        >>> h.calculate_event_likelihood(0, "A", 0.2, matrix, mu, 5)
        0.013333333333333332
        >>> h.calculate_event_likelihood(0, "D", 0.2, matrix, mu, 5)
        0.013333333333333332
        >>> h.calculate_event_likelihood(0, "E", 0.2, matrix, mu, 5)
        0.013333333333333332

        """
        total_rate = copy_rate_matrix.sum() + mutation_rate * self._states.shape[1]
        p_of_choosing_this_node = (
            copy_rate_matrix.sum(0)[node] + mutation_rate
        ) / total_rate
        odds_state = mutation_rate / nlang
        total_odds = mutation_rate
        for edgefrom, p in enumerate(copy_rate_matrix[:, node]):
            neighbor_state = self.before(edgefrom, time)
            if neighbor_state == new_state:
                odds_state += p
            total_odds += p
        return p_of_choosing_this_node * odds_state / total_odds

    def calculate_no_change_likelihood(
        self,
        time: float,
        copy_rate_matrix: numpy.ndarray,
        mutation_rate: float,
        nlang: int,
    ):
        """Calculate the probability for non-change event.

        This calculates the likelihood that an event, given to happen at a
        specific time point, is unobservabve because it does not change the
        state of the system. It does not compute the marginal likelihood of
        that change based on later changes.


        Example
        =======

        Imagine three equally connected nodes
        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        that start in a mixed state
        >>> start = ["A", "A", "B"]

        out of a total of 5 different states
        >>> nlang = 5

        and no history of change yet.
        >>> h = History(start, [])

        Mutations happen with half the rate of
        anything being copied.

        >>> mu = 0.5

        Then out of the 3*2.5 rate-weighted events, the ones that do not change
        the system are

         - Node 0 copies state A from node 1 (at rate 1),
         - Node 0 mutates to state A randomly (p=1/5 at rate 0.5)
        together at likelihood
        >>> h.calculate_event_likelihood(0, "A", 0.2, matrix, mu, nlang)
        0.14666666666666667

         - Node 1 copies state A from node 0 (at rate 1),
         - Node 1 mutates to state A randomly (p=1/5 at rate 0.5)
        together at likelihood
        >>> h.calculate_event_likelihood(1, "A", 0.2, matrix, mu, nlang)
        0.14666666666666667

        - Node 2 mutates to state B randomly (p=1/5 at rate 0.5)
        >>> h.calculate_event_likelihood(2, "B", 0.2, matrix, mu, nlang)
        0.013333333333333332

        Together, this is a likelihood of
        >>> h.calculate_no_change_likelihood(0.2, matrix, mu, nlang)
        0.30666666666666664

        For two connected nodes with identical values, the only way of change
        is a mutation. If there are only two values, each mutation is
        observable with probability 1/2. If mutation happens at the same rate
        as copying, the probability of an actual change is 1/4.

        >>> History(["A", "A"], []).calculate_no_change_likelihood(
        ...   0.5, numpy.array([[0,1],[1,0]]), 1, 2)
        0.75

        """
        p = 0
        for node in range(self._states.shape[1]):
            state_before = self.before(node, time)
            p += self.calculate_event_likelihood(
                node, state_before, time, copy_rate_matrix, mutation_rate, nlang
            )
        return p

    def calculate_change_likelihood(
        self,
        node: int,
        new_state: L,
        time: float,
        copy_rate_matrix: numpy.ndarray,
        mutation_rate: float,
        nlang: int,
    ):
        """Calculate the likelihood of a change at some point in the history.

        This only calculates the likelihood that at a specific time point, a
        particular node takes the given new value based on earlier events. It
        does not compute the marginal likelihood of that change based on later
        changes. The likelihood is conditioned on the event being observable as
        an actual change in the system state.

        Raises
        ======
        ValueError("Event is no change")
            if the ‘new’ node value is the same as the node value before the change.

        See also
        ========

        History.calculate_event_likelihood:
            The same likelihood calculation, not conditioned on the fact than an
            event is an actual change.
        History.calculate_no_change_likelihood:
            The likelihood of a no-change event, used for conditioning.

        Examples
        ========

        Imagine three equally connected nodes
        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])

        that start in two different states
        >>> start = ["A", "A", "B"]

        out of a total of 5 different states
        >>> nlang = 5

        and no history of change yet.
        >>> h = History(start, [])

        Mutations happen with half the rate of anything being copied.

        >>> mu = 0.5

        Then a first event (let us assume at time 0.2) that sets the new value
        of node 0 to A is no change.

        >>> h.calculate_change_likelihood(0, "A", 0.2, matrix, mu, nlang)
        Traceback (most recent call last):
          ...
        ValueError: Event is no change

        The only copy events that are observable are those where node 2 copies
        node 0 or node 1 (total rate 2) or vice versa (also total rate 2); in
        addition, 4 out of 5 mutation events for each node are observable, each
        with equal probability (rate 0.1 each). So node 2 gaining state A has a
        probability of (2+0.1)/(2+2+3*0.4) = 21/52
        >>> h.calculate_change_likelihood(2, "A", 0.2, matrix, mu, nlang) * 52
        20.999999999999996
        >>> h.calculate_change_likelihood(2, "C", 0.2, matrix, mu, nlang) * 52
        0.9999999999999999

        All the valid likelihoods obviously add up to 1.
        >>> p = 0
        >>> for node in 0,1,2:
        ...   for state in "A", "B", "C", "D", "E":
        ...     try:
        ...       p += h.calculate_change_likelihood(node, state, 0.2, matrix, mu, nlang)
        ...     except ValueError:
        ...       print(node, state)
        ...
        0 A
        1 A
        2 B
        >>> p
        1.0

        """
        if self.before(node, time) == new_state:
            raise ValueError("Event is no change")
        return self.calculate_event_likelihood(
            node, new_state, time, copy_rate_matrix, mutation_rate, nlang
        ) / (
            1
            - self.calculate_no_change_likelihood(
                time, copy_rate_matrix, mutation_rate, nlang
            )
        )

    def loglikelihood(self, end_time, copy_rate_matrix, mutation_rate, nlang):
        """Calculate the complete likelihood of the history.

        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> start = ["A", "A", "B"]
        >>> nlang = 5
        >>> mu = 0.5
        >>> h = History(start, [])

        With rates adding up to 7.5, of which 5.2 induce change, the
        probability of observing nothing is rather low (but not as low as taken
        when not conditioning on observable changes).

        The probability of seeing no event within 1 time unit when events have
        a total rate of 5.2 is

        >>> numpy.exp(-5.2)
        0.0055165644207607716

        so that should be (up to some rounding) the likelihood of still seeing
        an empty history after 1 time unit:

        >>> numpy.exp(h.loglikelihood(1.0, matrix, mu, nlang))
        0.0055165644207607716
        >>> numpy.allclose(
        ...   numpy.exp(h.loglikelihood(1.0, matrix, mu, nlang)),
        ...   numpy.exp(-5.2)
        ... )
        True

        Of course, if all starting values are identical and no mutations
        happen, the likelihood of nothing observable happening any more is 1.0.

        >>> constant = History(["A", "A", "A"], [])
        >>> numpy.exp(constant.loglikelihood(1.0, matrix, 0.0, 100))
        1.0

        If there are events, their likelihood is included. Consider a simple
        2-node model with mutation.

        >>> matrix = numpy.array([[0, 1], [1, 0]])
        >>> mu = 1
        >>> h = History(["A", "A"], [(0.5, 0, "B"), (1.0, 1, "B")])

        The probability to see this history after 2 time units is composed of

         - The probability that the first change happen at t=0.5. There are two
           possible change events, each happening with rate 0.5 (mutation rate
           1 and probability 1/2 of an actual change upon mutation), so this
           probability is exp(-0.5).

         - The probability that – given that a change does happen at t=0.5 –
           node 0 then flips to B. The only other possible change is that node
           1 flips to B, which happens at the same rate, so this has
           probability 1/2.

           >>> h.calculate_change_likelihood(0, "B", 0.5, matrix, mu, 2)
           0.5

        >>> numpy.exp(h.loglikelihood(0.5, matrix, mu, 2)) == numpy.exp(-0.5) * 0.5
        True

         - The probability that the second change happen at t=1.0. Now there
           are four possible observable change events, the two mutations (rate
           1/2 each) and node 0 copying from node 1 or vice versa (rate 1
           each). A waiting time of 1/2 given a rate of 3 has probability
           3*exp(-3*0.5).

         - The probability that, given this time of a change, the change is
           that node 1 also flips to B. Again symmetry shows that this has
           probability 1/2.

        >>> numpy.allclose(
        ...   numpy.exp(h.loglikelihood(1.0, matrix, mu, 2)),
        ...   numpy.exp(-0.5) * 0.5 * 3*numpy.exp(-1.5) * 0.5)
        True

         - The probability that in the new homogenous state (B, B) no change
           happens for another 1.0 time units, which has probabilty exp(-1).

        >>> numpy.allclose(
        ...   numpy.exp(h.loglikelihood(2.0, matrix, mu, 2)),
        ...   numpy.exp(-0.5) * 0.5 * 3*numpy.exp(-1.5) * 0.5 * numpy.exp(-1))
        True

        """
        prev_time = 0.0
        logp = 0.0
        state = self._states[0].copy()
        observable_transition = state[None, :] != state[:, None]
        mutation_change_rate = (
            mutation_rate * self._states.shape[1] * (nlang - 1) / nlang
        )
        rate_of_observable_change = (
            copy_rate_matrix * observable_transition
        ).sum() + mutation_change_rate
        for time, node, value in self.all_changes():
            if time > end_time:
                break
            # Contribution from waiting time
            logp += expon(scale=1 / rate_of_observable_change).logpdf(time - prev_time)
            prev_time = time

            # Contribution from change
            logp += numpy.log(
                self.calculate_change_likelihood(
                    node, value, time, copy_rate_matrix, mutation_rate, nlang
                )
            )
            state[node] = value
            observable_transition[node, :] = state != value
            observable_transition[:, node] = state != value
            rate_of_observable_change = (
                copy_rate_matrix * observable_transition
            ).sum() + mutation_change_rate
        return logp + expon(scale=1 / rate_of_observable_change).logsf(
            end_time - prev_time
        )

# GOALS:
# (a) A Gibbs operator that changes the value intervall between two subsequent changes of one node, based on all changes on adjacent nodes that interact with this change [ie. all that have either the old or the new value]
# (b) A Gibbs operator that changes the position of a change between the change before and the change after. This includes creating a change, from sliding it in from before the start/after the endof the observed interval, or deleting a change, from sliding it out of the observed interval.
# (c) A pair of reversible jump operators which split a change into two or merge two adjacent changes into one.

def add_pair_of_changes(history):
    ...
def split_change(history):
    ...
def merge_changes(history):
    ...
def gibbs_redraw_change(history: History):
    all_changes = history.all_changes()
    time, node, value = all_changes[numpy.random.randint(len(all_changes))]
    cump = 0
    for i in range(nlang):
        try:
            p_i = history.calculate_change_likelihood(node, i, time, matrix, mu, nlang)
        except ValueError:
            p_i = 0.0
        # FIXME missing: how does this affect subsequent changes?
        cump.append(p_i + cump[-1])
    new_value = bisect.bisect(cump[1:], numpy.random.random())

def slide_change(history):
    ...

def redraw_change(history: history):
    ...

def select_operator():

end = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
end_time = 1
matrix = numpy.diag([1 for _ in end[1:]], 1) + numpy.diag([1 for _ in end[1:]], -1)
nlang = 200
history = History(end, [])
mu = 1e-4
log_likelihood = history.loglikelihood(end_time, matrix, mu, nlang)
while True:
    operator = select_operator()
    candidate_history = operator(deepcopy.deepcopy(history))
    candidate_log_likelihood = candidate_history.loglikelihood(
        end_time, matrix, mu, nlang
    )
    if candidate_log_likelihood > log_likelihood:
        history = candidate_history
    else:
        p = numpy.exp(candidate_log_likelihood - log_likelihood)
        if numpy.random.random() < p:
            history = candidate_history


if __name__ == "__main__":
    start = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])

    copy_rate_matrix = numpy.diag([1 for _ in start[1:]], 1) + numpy.diag(
        [1 for _ in start[1:]], -1
    )
    mutation_rate = 1e-3

    image = start.copy()[None, :]

    NLANG = 200

    def generate_history(start, end_time, rate_matrix):
        time = 0
        ps = numpy.cumsum(rate_matrix.flat)
        total_rate = ps[-1] + mutation_rate * len(start)
        time += numpy.random.exponential(1 / total_rate)
        while time < end_time:
            try:
                x = numpy.random.random() * total_rate
                edgefrom, edgeto = numpy.unravel_index(
                    bisect.bisect(ps, x), rate_matrix.shape
                )
                if start[edgeto] == start[edgefrom]:
                    continue
                start[edgeto] = start[edgefrom]
            except ValueError:
                lang = numpy.random.randint(NLANG)
                edgeto = numpy.random.randint(len(start))
                if start[edgeto] == lang:
                    continue
                start[edgeto] = lang

            yield time, edgeto, start[edgeto]
            time += numpy.random.exponential(1 / total_rate)

    for i in generate_history(start, 100, copy_rate_matrix):
        image = numpy.vstack((image, start[None, :]))

    plt.imshow(image.T, aspect="auto", interpolation="nearest")
    plt.show()
