import bisect
import itertools
import typing
from copy import deepcopy

import numpy
from matplotlib import pyplot as plt
from scipy.stats import expon

L = typing.TypeVar("L", bound=typing.Hashable)


class History(typing.Generic[L]):
    end: float

    def __init__(
        self,
        start: typing.Sequence[L],
        changes: typing.Iterable[tuple[float, int, L]],
        end: typing.Optional[float] = None,
    ) -> None:
        """Store a history.

        A history represents a starting state at time t=0.0, and a set of
        changes (t, n, v), where at time t the node at index n takes the new
        value v.

        The times are expected to be sorted; there is undefined behaviour when
        they are not.

        All changes must be actual changes, a ValueError is raised otherwise.

        >>> History([0, 0], [(0.5, 0, 0)], 1.0)
        Traceback (most recent call last):
          ...
        ValueError: Event is no change

        """
        start = numpy.asarray(start)
        self._states = start[None, :]
        self._times = numpy.zeros_like(self._states, dtype=float)
        history_depth = {}
        time = 0.0
        for time, node, new_value in changes:
            try:
                history_depth[node] += 1
            except KeyError:
                history_depth[node] = 1
            if new_value == self._states[history_depth[node] - 1, node]:
                raise ValueError("Event is no change")
            if history_depth[node] >= self._times.shape[0] - 1:
                self._times = numpy.vstack(
                    (self._times, numpy.full_like(self._times[0], numpy.inf))
                )
                self._states = numpy.vstack(
                    (self._states, numpy.zeros_like(self._states[0]))
                )
            self._times[history_depth[node], node] = time
            self._states[history_depth[node], node] = new_value
        if end is None:
            end = time
        if end < time:
            raise ValueError("End time before the last event.")
        self.end = end

    def __repr__(self):
        return f"History({self.start()!r}, {self.all_changes()}, {self.end})"

    def all_changes(self) -> typing.Iterable[tuple[float, int, L]]:
        """List all changes documented in this history.

        Extract the times, nodes, and values of changes, similar to the
        iterable given to the constructor.

        >>> random_times = numpy.cumsum(numpy.random.random(10))
        >>> random_nodes = numpy.random.randint(3, size=10)
        >>> random_values = numpy.random.randint(300, size=10)
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
                zip(self._times.flat, self._states.flat)
            )
            if 0 < time <= self.end
        )

    def start(self) -> numpy.ndarray:
        """Return the start state."""
        view = self._states[0]
        view.flags.writeable = False
        return view

    def related(
        self, time: float, node: int, connections: typing.Iterable[int]
    ) -> typing.Iterator[tuple[float, int, L]]:
        """Find all the changes that are related to the state of this node at this time.

        Given a node, find the time interval where this node has this value,
        and then return all other changes as (time, node, new value) triples
        where the node is connected to this node according to the connections
        and the time is in the interval. (If the time given matches a time of
        change, it is taken as the start of the interval, as usual.)

        Examples
        ========

        >>> h = History(
        ...   numpy.array(["A", "A", "B"], dtype="<U1"),
        ...   [(0.05, 2, "A"),
        ...    (0.1, 0, "C"),
        ...    (0.25, 1, "C"),
        ...    (0.3, 2, "B"),
        ...    (0.4, 0, "B"),
        ...    (0.5, 1, "B")],
        ...    0.6)
        >>> list(h.related(0.05, 0, [1, 2]))
        [(0.05, 2, 'A')]
        >>> list(h.related(0.15, 0, [1, 2]))
        [(0.25, 1, 'C'), (0.3, 2, 'B')]
        >>> list(h.related(0.1, 0, [1, 2])) == _
        True
        >>> list(h.related(0.45, 0, [1, 2]))
        [(0.5, 1, 'B')]
        >>> list(h.related(0.55, 1, [0, 2]))
        []

        NOTE: The changes are returned in node order (as given by
        `connections`), then for each node in time order. To get an overall
        time order, use `sorted()` etc.

        >>> list(h.related(0.4, 1, [0, 2]))
        [(0.4, 0, 'B'), (0.3, 2, 'B')]

        """
        time_slice = bisect.bisect(self._times[:, node], time)
        start = self._times[time_slice - 1, node]
        try:
            end = self._times[time_slice, node]
        except IndexError:
            end = self.end
        if end > self.end:
            end = self.end
        for other_node in connections:
            istart = bisect.bisect(self._times[:, other_node], start)
            iend = bisect.bisect(self._times[:, other_node], end)
            if istart == iend:
                # No change to that node in this interval
                continue
            for i in range(istart, iend):
                if self._times[i, other_node] > self.end:
                    continue
                try:
                    yield self._times[i, other_node], other_node, self._states[
                        i, other_node
                    ]
                except IndexError:
                    continue

    def at(self, time: float, node: int) -> L:
        """Look up the value of a node at a particular time.

        Examples
        ========

        >>> h = History(
        ...   numpy.array(["A", "A"], dtype="<U1"),
        ...   [(0.1, 0, "C"), (0.25, 1, "C"), (0.4, 0, "B")])

        So at the start, every node has value A.

        >>> h.at(0.0, 0)
        'A'
        >>> h.at(0.0, 1)
        'A'

        At time 0.3, both nodes have changed to C.

        >>> h.at(0.3, 0)
        'C'
        >>> h.at(0.3, 1)
        'C'

        For the exact time of a change, the function reports the new value.

        >>> h.at(0.25, 1)
        'C'


        In the end, the nodes have changed to B and C.

        >>> h.at(999., 0)
        'B'
        >>> h.at(999., 1)
        'C'

        Going back before the start is also possible, and just returns the
        starting values.

        >>> h.at(-999., 0)
        'A'
        >>> h.at(-999., 1)
        'A'
        """
        time_slice = bisect.bisect_right(self._times[:, node], time) - 1
        if time_slice == -1:
            time_slice = 0
        return self._states[time_slice, node]

    def before(self, time: float, node: int) -> L:
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
        ...     (0.0, 0), (0.0, 1), (0.3, 0), (0.3, 1),
        ...     (999., 0), (999., 1), (-999., 0), (-999., 1)]:
        ...   assert h.at(node, time) == h.before(node, time)

        For the exact time of a change, the function reports the old value.

        >>> h.before(0.1, 0)
        'A'
        >>> h.before(0.25, 1)
        'A'
        >>> h.before(0.4, 0)
        'C'

        """
        time_slice = bisect.bisect_left(self._times[:, node], time) - 1
        if time_slice == -1:
            time_slice = 0
        return self._states[time_slice, node]

    def calculate_event_likelihood(
        self,
        time: float,
        node: int,
        new_state: L,
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

        >>> h.calculate_event_likelihood(0.2, 0, "B", matrix, 0.0, 5)
        0.16666666666666666
        >>> h.calculate_event_likelihood(0.2, 1, "A", matrix, 0.0, 5)
        0.16666666666666666

        No other change has positive probability.
        >>> for i in "A", "D", "E":
        ...   print(h.calculate_event_likelihood(0.2, 0, "D", matrix, 0.0, 5))
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

        >>> h.calculate_event_likelihood(0.2, 0, "B", matrix, mu, 5)
        0.14666666666666667
        >>> h.calculate_event_likelihood(0.2, 0, "C", matrix, mu, 5)
        0.14666666666666667
        >>> h.calculate_event_likelihood(0.2, 0, "A", matrix, mu, 5)
        0.013333333333333332
        >>> h.calculate_event_likelihood(0.2, 0, "D", matrix, mu, 5)
        0.013333333333333332
        >>> h.calculate_event_likelihood(0.2, 0, "E", matrix, mu, 5)
        0.013333333333333332

        """
        total_rate = copy_rate_matrix.sum() + mutation_rate * self._states.shape[1]
        p_of_choosing_this_node = (
            copy_rate_matrix.sum(0)[node] + mutation_rate
        ) / total_rate
        odds_state = mutation_rate / nlang
        total_odds = mutation_rate
        for edgefrom, p in enumerate(copy_rate_matrix[:, node]):
            neighbor_state = self.before(time, edgefrom)
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
        >>> h.calculate_event_likelihood(0.2, 0, "A", matrix, mu, nlang)
        0.14666666666666667

         - Node 1 copies state A from node 0 (at rate 1),
         - Node 1 mutates to state A randomly (p=1/5 at rate 0.5)
        together at likelihood
        >>> h.calculate_event_likelihood(0.2, 1, "A", matrix, mu, nlang)
        0.14666666666666667

        - Node 2 mutates to state B randomly (p=1/5 at rate 0.5)
        >>> h.calculate_event_likelihood(0.2, 2, "B", matrix, mu, nlang)
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
            state_before = self.before(time, node)
            p += self.calculate_event_likelihood(
                time, node, state_before, copy_rate_matrix, mutation_rate, nlang
            )
        return p

    def calculate_change_likelihood(
        self,
        time: float,
        node: int,
        new_state: L,
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

        >>> h.calculate_change_likelihood(0.2, 0, "A", matrix, mu, nlang)
        Traceback (most recent call last):
          ...
        ValueError: Event is no change

        The only copy events that are observable are those where node 2 copies
        node 0 or node 1 (total rate 2) or vice versa (also total rate 2); in
        addition, 4 out of 5 mutation events for each node are observable, each
        with equal probability (rate 0.1 each). So node 2 gaining state A has a
        probability of (2+0.1)/(2+2+3*0.4) = 21/52
        >>> h.calculate_change_likelihood(0.2, 2, "A", matrix, mu, nlang) * 52
        20.999999999999996
        >>> h.calculate_change_likelihood(0.2, 2, "C", matrix, mu, nlang) * 52
        0.9999999999999999

        All the valid likelihoods obviously add up to 1.
        >>> p = 0
        >>> for node in 0,1,2:
        ...   for state in "A", "B", "C", "D", "E":
        ...     try:
        ...       p += h.calculate_change_likelihood(0.2, node, state, matrix, mu, nlang)
        ...     except ValueError:
        ...       print(node, state)
        ...
        0 A
        1 A
        2 B
        >>> p
        1.0

        """
        if self.before(time, node) == new_state:
            raise ValueError("Event is no change")
        return self.calculate_event_likelihood(
            time, node, new_state, copy_rate_matrix, mutation_rate, nlang
        ) / (
            1
            - self.calculate_no_change_likelihood(
                time, copy_rate_matrix, mutation_rate, nlang
            )
        )

    def loglikelihood(
        self, copy_rate_matrix, mutation_rate, nlang, start=0.0, end=None
    ):
        """Calculate the complete likelihood of the history.

        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> start = ["A", "A", "B"]
        >>> nlang = 5
        >>> mu = 0.5
        >>> h = History(start, [], 1.0)

        With rates adding up to 7.5, of which 5.2 induce change, the
        probability of observing nothing is rather low (but not as low as taken
        when not conditioning on observable changes).

        The probability of seeing no event within 1 time unit when events have
        a total rate of 5.2 is

        >>> numpy.exp(-5.2)
        0.0055165644207607716

        so that should be (up to some rounding) the likelihood of still seeing
        an empty history after 1 time unit:

        >>> numpy.exp(h.loglikelihood(matrix, mu, nlang))
        0.0055165644207607716
        >>> numpy.allclose(
        ...   numpy.exp(h.loglikelihood(matrix, mu, nlang)),
        ...   numpy.exp(-5.2)
        ... )
        True

        Of course, if all starting values are identical and no mutations
        happen, the likelihood of nothing observable happening any more is 1.0.

        >>> constant = History(["A", "A", "A"], [], 1.0)
        >>> numpy.exp(constant.loglikelihood(matrix, 0.0, 100))
        1.0

        If there are events, their likelihood is included. Consider a simple
        2-node model with mutation.

        >>> matrix = numpy.array([[0, 1], [1, 0]])
        >>> mu = 1
        >>> h = History(["A", "A"], [(0.5, 0, "B"), (1.0, 1, "B")], end=2.0)

        The probability to see this history is composed of

         - The probability that the first change happen at t=0.5. There are two
           possible change events, each happening with rate 0.5 (mutation rate
           1 and probability 1/2 of an actual change upon mutation), so this
           probability is exp(-0.5).

         - The probability that – given that a change does happen at t=0.5 –
           node 0 then flips to B. The only other possible change is that node
           1 flips to B, which happens at the same rate, so this has
           probability 1/2.

           >>> h.calculate_change_likelihood(0.5, 0, "B", matrix, mu, 2)
           0.5

        >>> numpy.exp(h.loglikelihood(matrix, mu, 2, end=0.5)) == numpy.exp(-0.5) * 0.5
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
        ...   numpy.exp(h.loglikelihood(matrix, mu, 2, start=0.5, end=1.0)),
        ...   3*numpy.exp(-1.5) * 0.5)
        True

         - The probability that in the new homogenous state (B, B) no change
           happens for another 1.0 time units, which has probabilty exp(-1).

        >>> numpy.allclose(
        ...   numpy.exp(h.loglikelihood(matrix, mu, 2, start=1.0)),
        ...   numpy.exp(-1))
        True
        >>> numpy.allclose(
        ...   numpy.exp(h.loglikelihood(matrix, mu, 2)),
        ...   numpy.exp(-0.5) * 0.5 * 3*numpy.exp(-1.5) * 0.5 * numpy.exp(-1))
        True

        >>> numpy.allclose(
        ...   h.loglikelihood(matrix, mu, 2, end=0.5) +
        ...   h.loglikelihood(matrix, mu, 2, start=0.5, end=1.0) +
        ...   h.loglikelihood(matrix, mu, 2, start=1.0),
        ...   h.loglikelihood(matrix, mu, 2))
        True

        """
        if end is None:
            end = self.end
        prev_time = start
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
            if time > end:
                break
            # Contribution from waiting time
            if time > start:
                logp += expon(scale=1 / rate_of_observable_change).logpdf(
                    time - prev_time
                ) + numpy.log(
                    self.calculate_change_likelihood(
                        time, node, value, copy_rate_matrix, mutation_rate, nlang
                    )
                )
                prev_time = time

            # Contribution from change
            state[node] = value
            observable_transition[node, :] = state != value
            observable_transition[:, node] = state != value
            rate_of_observable_change = (
                copy_rate_matrix * observable_transition
            ).sum() + mutation_change_rate
        return logp + expon(scale=1 / rate_of_observable_change).logsf(end - prev_time)

    def alternatives_with_likelihood(
        self,
        time: float,
        node: int,
        copy_rate_matrix: numpy.ndarray,
        mutation_rate: float,
        nlang: int,
    ) -> typing.Iterable[tuple[int, float]]:
        """Calculate the possible alternatives for the specified change.

        In a history where a given node changes at a given time, compute the
        probability of each value the node could take, given the other parts of the
        history.

        """
        alternative_history = deepcopy(self)
        time_slice = bisect.bisect_right(alternative_history._times[:, node], time) - 1
        is_neighbor = (copy_rate_matrix[:, node] > 0) | (copy_rate_matrix[node, :] > 0)
        neighbors = numpy.arange(len(is_neighbor))[is_neighbor]
        dependent_changes = list(self.related(time, node, neighbors))
        forbidden_values = set()
        try:
            next_time = self._times[time_slice + 1, node]
            if next_time > self.end:
                raise IndexError
            forbidden_values.add(
                self._states[time_slice + 1, node],
            )
        except IndexError:
            next_time = self.end
        if time_slice > 0:
            forbidden_values.add(self._states[time_slice - 1, node])
        nontrivial_values = (
            {v for _, _, v in dependent_changes}
            # TODO: Technically, only the incoming neighbors are relevant
            # ‘before’, not the outgoing neighbors.
            | {self.before(time, neighbor) for neighbor in neighbors}
        ) - forbidden_values
        # Well, that's a lie. The ‘trivial’ values are not exactly trivial to
        # compute. But they all have the same likelihood, so they can all be
        # handled the same.
        trivial_values = set(range(nlang)) - nontrivial_values - forbidden_values

        some_trivial_value = trivial_values.pop()
        for value in itertools.chain(nontrivial_values, [some_trivial_value]):
            likelihood = 1.0
            alternative_history._states[time_slice, node] = value
            loglikelihood = alternative_history.loglikelihood(
                copy_rate_matrix,
                mutation_rate,
                nlang,
                numpy.nextafter(time, -numpy.inf),
                next_time,
            )
            # There should be a way to compute the likelihood change
            # considering only the connected nodes. But the adjustment for the
            # conditional probability that no change happens, given the assumed
            # state of node, is a bit more difficult than I thought.
            for d_time, d_node, d_value in dependent_changes:
                # if d_value is the same as value, then the probability of
                # seeing no change in either node or d_node goes up by the
                # connection strength.

                # if d_value is different from value, then the probability of
                # seeing no change in either goes down by the connection
                # strength.

                # But I don't know whether that change has a simple expression
                # in terms of expon, something like this.
                likelihood *= expon(
                    scale=1
                    / (copy_rate_matrix[d_node, node] + copy_rate_matrix[node, d_node])
                ).pdf(d_time - time)
                likelihood *= alternative_history.calculate_change_likelihood(
                    d_time, d_node, d_value, copy_rate_matrix, mutation_rate, nlang
                )
            yield value, numpy.exp(loglikelihood)
        for value in trivial_values:
            yield value, numpy.exp(loglikelihood)


# GOALS:
# (b) A Gibbs operator that changes the position of a change between the change before and the change after. This includes creating a change, from sliding it in from before the start/after the endof the observed interval, or deleting a change, from sliding it out of the observed interval.
# (c) A pair of reversible jump operators which split a change into two or merge two adjacent changes into one.


def gibbs_redraw_change(history, change=None):
    """A Gibbs operator changing the value of a change.

    A Gibbs operator that changes the value on the interval between two
    subsequent changes of one node, based on all changes on adjacent nodes that
    interact with this change.

    """
    changes = history.all_changes()
    if change is None:
        try:
            change = numpy.random.randint(len(changes))
        except ValueError:
            return -numpy.inf, None
    time, node, value = changes[change]
    sump = 0.0
    probabilities = []
    values = []
    for alternative, probability in history.alternatives_with_likelihood(
        time,
        node,
        copy_rate_matrix,
        mutation_rate,
        nlang,
    ):
        sump += probability
        probabilities.append(sump)
        values.append(alternative)
    if sump == 0:
        new_value = values[numpy.random.randint(len(values))]
    else:
        new_value = values[bisect.bisect(probabilities, numpy.random.random() * sump)]
    if new_value == value:
        return numpy.inf, history
    else:
        return numpy.inf, History(
            history.start(),
            [
                (t, n, v if j != change else new_value)
                for j, (t, n, v) in enumerate(history.all_changes())
            ],
            history.end,
        )


def gibbs_redraw_start(history, node=None):
    """A Gibbs operator changing the value of a change.

    A Gibbs operator that changes the value on the interval between two
    subsequent changes of one node, based on all changes on adjacent nodes that
    interact with this change.

    """
    start = list(history.start())
    if node is None:
        node = numpy.random.randint(len(start))
    value = start[node]
    sump = 0.0
    probabilities = []
    values = []
    for alternative, probability in history.alternatives_with_likelihood(
        0.0,
        node,
        copy_rate_matrix,
        mutation_rate,
        nlang,
    ):
        sump += probability
        probabilities.append(sump)
        values.append(alternative)
    if sump == 0:
        new_value = values[numpy.random.randint(len(values))]
    else:
        new_value = values[bisect.bisect(probabilities, numpy.random.random() * sump)]
    if new_value == value:
        return numpy.inf, history
    else:
        start[node] = new_value
        return numpy.inf, History(
            start,
            history.all_changes(),
            history.end,
        )


def move_change(history):
    actual_changes = (history._times <= history.end).sum(0) - 1
    heap = actual_changes.cumsum()
    if heap[-1] == 0:
        return -numpy.inf, history
    random_node = bisect.bisect(heap, numpy.random.randint(heap[-1]))
    random_change = numpy.random.randint(1, actual_changes[random_node] + 1)
    lower = history._times[random_change - 1, random_node]
    try:
        upper = history._times[random_change + 1, random_node]
    except IndexError:
        upper = history.end
    if upper > history.end:
        upper = history.end
    alternative_history = deepcopy(history)
    alternative_history._times[random_change, random_node] = numpy.random.uniform(
        lower, upper
    )
    return 0.0, alternative_history


def split_change(history):
    # Pick a random change, including for each node the one before the start
    # and the one after the end of history
    n_fict_changes = (history._times <= history.end).sum(0) + 1
    heap = n_fict_changes.cumsum()
    random_node = bisect.bisect(heap, numpy.random.randint(heap[-1]))
    if n_fict_changes[random_node] <= 2:
        print("Case: No change")
        lower = numpy.random.uniform(-expon().rvs(), 0)
        upper = numpy.random.uniform(history.end, history.end + expon().rvs())
        mid = numpy.random.uniform(lower, upper - history.end)
        if mid > 0:
            mid += history.end
            value_of_second_change = numpy.random.randint(nlang - 1)
            if value_of_second_change >= history._states[0, random_node]:
                value_of_second_change += 1
        else:
            value_of_second_change = history._states[0, random_node]
    else:
        random_change = numpy.random.randint(0, n_fict_changes[random_node] + 1)
        if random_change == 0:
            print("Case: Change before the start of history")
            # Potentially moving it into the history.
            lower = numpy.random.uniform(-expon().rvs(), 0)
            upper = history._times[random_change + 1, random_node]
            if upper > history.end:
                upper = numpy.random.uniform(history.end, history.end + expon().rvs())
            mid = numpy.random.uniform(lower, 0.0)
            value_of_second_change = history._states[0, random_node]
        elif random_change == n_fict_changes[random_node]:
            print("Case: Change after the end of history")
            # Potentially adding the new change before the end.
            lower = history._times[random_change - 1, random_node]
            upper = numpy.random.uniform(history.end, history.end + expon().rvs())
            mid = numpy.random.uniform(history.end, upper)
            value_of_second_change = history.at(history.end, random_node)
        else:
            print("Case: DEFAULT – Change in history")
            lower = history._times[random_change - 1, random_node]
            upper = history._times[random_change + 1, random_node]
            if upper > history.end:
                upper = numpy.random.uniform(history.end, history.end + expon().rvs())
            mid = history._times[random_change, random_node]
            value_of_second_change = history._states[random_change, random_node]

    new_change_time = numpy.random.uniform(lower, mid)
    move_existing_change_to = numpy.random.uniform(mid, upper)
    if move_existing_change_to <= 0.0:
        print("Sub 0")
        # The old change was before history, and the second of the new changes
        # is also before history. History doesn't change.
        return 0.0, history
    elif move_existing_change_to > history.end:
        if new_change_time > history.end:
            print("Sub 1")
            # The new change is outside history, so the moved old change even
            # more so. History doesn't change.
            return 0.0, history
        if new_change_time <= 0.0:
            print("Sub 2")
            # Both the change before and the change after are on different ends
            # of history, so change the value of the node – completely
            # throughout history – at random
            _, alternative_history = gibbs_redraw_start(history, node=random_node)
            return 0.0, alternative_history
        else:
            print("Sub 3")
            # Add a new change at the end
            changes = history.all_changes()
            new_change_pos = bisect.bisect(changes, (new_change_time, random_node, -1))
            changes.insert(new_change_pos, (new_change_time, random_node, -1))
            alternative_history = History(history.start(), changes, history.end)
            _, alternative_history = gibbs_redraw_change(
                alternative_history, change=new_change_pos
            )
            return 0.0, alternative_history
    else:
        if mid <= 0.0:
            # The second change is inside history, but it wasn't before.
            print("Sub 4a")
            # Add the new change before the start, i.e. change the starting value and the position of the first change.
            alternative_history = deepcopy(history)
            new_value = numpy.random.randint(nlang - 1)
            if new_value >= value_of_second_change:
                new_value += 1
            if alternative_history._states[-1, random_node] != numpy.inf:
                # This node is one of those with the longest history.
                # To add another change for it, we need to first expand the history array.
                alternative_history._times = numpy.vstack(
                    (
                        alternative_history._times,
                        numpy.full_like(alternative_history._times[0], numpy.inf),
                    )
                )
                alternative_history._states = numpy.vstack(
                    (
                        alternative_history._states,
                        numpy.zeros_like(alternative_history._states[0]),
                    )
                )
                alternative_history._states[1, random_node] = new_value
            alternative_history._states[0, random_node] = new_value
            alternative_history._states[1, random_node] = value_of_second_change
            alternative_history._times[1, random_node] = move_existing_change_to
            return 0.0, alternative_history
        elif new_change_time <= 0.0:
            print("Sub 4")
            # Add the new change before the start, i.e. change the starting value and the position of the first change.
            alternative_history = deepcopy(history)
            new_value = numpy.random.randint(nlang - 1)
            if new_value >= value_of_second_change:
                new_value += 1
            alternative_history._states[0, random_node] = new_value
            alternative_history._states[1, random_node] = value_of_second_change
            alternative_history._times[1, random_node] = move_existing_change_to
            return 0.0, alternative_history
        else:
            print("Sub 5")
            # The normal case: The new change and the moved change are both within the history.
            changes = [
                (t, n, v)
                for t, n, v in history.all_changes()
                if t != mid or n != random_node or v != value_of_second_change
            ]
            new_change_pos = bisect.bisect(changes, (new_change_time, random_node, 0))
            changes.insert(new_change_pos, (new_change_time, random_node, -1))
            moved_change_pos = bisect.bisect(
                changes, (move_existing_change_to, random_node, value_of_second_change)
            )
            changes.insert(
                moved_change_pos,
                (move_existing_change_to, random_node, value_of_second_change),
            )

            alternative_history = History(history.start(), changes, history.end)
            _, alternative_history = gibbs_redraw_change(
                alternative_history, change=new_change_pos
            )
            return 0.0, alternative_history


def merge_changes(history):
    # Pick a pair of subsequent random changes (which can lie outside the history)
    n_fict_changes = (history._times <= history.end).sum(0)
    heap = n_fict_changes.cumsum()
    random_node = bisect.bisect(heap, numpy.random.randint(heap[-1]))
    random_change = numpy.random.randint(n_fict_changes[random_node])
    if 0 == n_fict_changes[random_node] - 1:
        # There is no actual change on this segment.
        ...
    elif random_change == 0:
        # Merge the first change with the change before the start. This can
        # either remove the first change, or it can change the start value and
        # move the first change.
        before_start = -expon().rvs()
        mid = (before_start + history._times[random_change + 1, random_node]) / 2.0
        if mid < 0.0:
            # Merged to before beginning: Remove change and set its value as the starting value
            history = deepcopy(history)
            history._states[0:-1, random_node] = history._states[1:, random_node]
            history._times[1:-1, random_node] = history._states[2:, random_node]
            history._times[0, random_node] = 0
            history._times[-1, random_node] = numpy.inf
            return 0.0, history
        else:
            history._times[1, random_node]
            ...
    elif random_change == n_fict_changes[random_node] - 1:
        # Merge the last change with the change after the end of history. This
        # can either remove the last change (if it is merged to after the end
        # of history), or it can change the value and move that last change.
        after_end = history.end + expon().rvs()
        mid = (history._times[random_change, random_node] + after_end) / 2.0
        if end > history.end:
            ...
        else:
            ...
    else:
        # Both changes lie inside history. Check that they are different,
        # otherwise the merge will not be a change (so the un-merge can not be
        # generated by split_change, and accepting this move would violate the
        # reversible jump assumption.)
        history = deepcopy(history)
        value_before = history._states[random_change - 1, random_node]
        value_after = history._states[random_change + 1, random_node]
        if value_before == value_after:
            return -numpy.inf, history
        mid = (
            history._times[random_change, random_node]
            + history._times[random_change + 1, random_node]
        ) / 2.0
        history._states[random_change:-1, random_node] = history._states[
            random_change + 1 :, random_node
        ]
        history._times[random_change + 1 : -1, random_node] = history._states[
            random_change + 2 : -1, random_node
        ]
        history._times[random_change, random_node] = mid
        history._times[-1, random_node] = numpy.inf
        return 0.0, history


def add_change(history):
    pos = numpy.random.uniform(0.0, history.end)
    random_node = numpy.random.randint(len(history.start()))
    new_value = numpy.random.randint(nlang - 2)
    if new_value >= history.at(pos, random_node):
        new_value += 1
    changes = history.all_changes()
    bisect.insort(
            changes,
            (pos, random_node, new_value),
        )
    try:
        return numpy.inf, History(history.start(), changes, history.end)
    except ValueError:
        return -numpy.inf, history

def remove_change(history):
    # Pick a pair of subsequent random changes
    n_changes = (history._times <= history.end).sum(0) - 1
    heap = n_changes.cumsum()
    if heap[-1] == 0:
        return -numpy.inf, history
    random_node = bisect.bisect(heap, numpy.random.randint(heap[-1]))
    random_change = numpy.random.randint(1, n_changes[random_node] + 1)

    changes = history.all_changes()
    changes.remove(
        (
            history._times[random_change, random_node],
            random_node,
            history._states[random_change, random_node],
        )
    )
    try:
        return 0.0, History(history.start(), changes, history.end)
    except ValueError:
        return -numpy.inf, history


def add_or_remove_change(history):
    if numpy.random.randint(2):
        return add_change(history)
    else:
        return remove_change(history)


def select_operator():
    all_operators = [add_or_remove_change, gibbs_redraw_change, gibbs_redraw_start] + [
        move_change
    ] * 2
    return all_operators[numpy.random.randint(len(all_operators))]


# if __name__ == "__main__":
if True:
    end = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
    end_time = 100.0
    copy_rate_matrix = numpy.diag([1 for _ in end[1:]], 1) + numpy.diag(
        [1 for _ in end[1:]], -1
    )
    nlang = 200
    history = History(end, [(50, 10, 2)], end=end_time)
    mutation_rate = 1e-4
    log_likelihood = history.loglikelihood(copy_rate_matrix, mutation_rate, nlang)
    while True:
        operator = select_operator()
        hastings_ratio, candidate_history = operator(deepcopy(history))
        if hastings_ratio == -numpy.inf:
            print(operator.__name__, "rejected: Impossible")
            continue
        candidate_log_likelihood = candidate_history.loglikelihood(
            copy_rate_matrix, mutation_rate, nlang
        )
        logp = candidate_log_likelihood + hastings_ratio - log_likelihood
        if (logp > 0) or (numpy.random.random() < numpy.exp(logp)):
            print(operator.__name__, "accepted")
            history = candidate_history
            print(history)
        else:
            print(operator.__name__, "rejected")

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
