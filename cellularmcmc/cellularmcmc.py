"""Methods for MCMC on a CTMC copy graph."""
import bisect
import itertools
import typing
from copy import deepcopy
from dataclasses import dataclass

import numpy
from scipy.stats import expon

L = typing.TypeVar("L", bound=typing.Hashable)


exponential_distribution = expon()


class History(typing.Generic[L]):
    """The history of a CTMC copy graph.

    A history represents a starting state at time t=0.0, and a set of
    changes (t, n, v), where at time t the node at index n takes the new
    value v.

    """

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

    def __eq__(self, o):
        valid = numpy.isfinite(self._times)
        o_valid = numpy.isfinite(o._times)
        return (self._states[valid] == o._states[o_valid]).all() and (
            self._times[valid] == o._times[o_valid]
        ).all()

    def __repr__(self):
        return f"History({self.start()!r}, {self.all_changes()}, {self.end})"

    def all_changes(self) -> typing.Iterable[tuple[float, int, L]]:
        """List all changes documented in this history.

        Extract the times, nodes, and values of changes, similar to the
        iterable given to the constructor.

        >>> random_times = numpy.cumsum(numpy.random.random(10))
        >>> random_nodes = numpy.random.randint(3, size=10)
        >>> random_values = numpy.arange(1, 10+1)
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

    def after(self, time: float, node: int) -> typing.Optional[L]:
        """Look up the next value a node will take at a particular time.

        Examples
        ========

        >>> h = History(
        ...   numpy.array(["A", "A"], dtype="<U1"),
        ...   [(0.1, 0, "C"), (0.25, 1, "C"), (0.4, 0, "B")])

        >>> h.after(0.0, 0)
        'C'
        >>> h.after(0.0, 1)
        'C'
        >>> h.after(0.3, 0)
        'B'
        >>> h.after(0.3, 1)
        >>> h.after(0.25, 1)
        >>> h.after(999., 0)
        >>> h.after(999., 1)
        >>> h.after(-999., 0)
        'A'
        >>> h.after(-999., 1)
        'A'
        """
        time_slice = bisect.bisect_right(self._times[:, node], time) - 1
        try:
            if self._times[time_slice + 1, node] > self.end:
                return None
        except IndexError:
            return None
        return self._states[time_slice + 1, node]

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


@dataclass
class HistoryModel:
    """A stochastic model of history."""

    copy_rate_matrix: numpy.ndarray
    mutation_rate: float
    languages: typing.Sequence[L]

    @property
    def nlang(self):
        """Derive the number of languages."""
        return len(self.languages)

    def calculate_event_likelihood(
        self,
        history: History[L],
        time: float,
        node: int,
        new_state: L,
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

        >>> h = History(start, [], end=1.0)
        >>> m = HistoryModel(matrix, 0.0, range(nlang))

        If the first change is at time 0.2, it will have equal probability of
        changing node 0 to B (copied from node 1) or C (copied from node 2);
        changing node 1 to A or C, or changing node 2 to A or B ??? so each has
        probability 1/6.

        >>> m.calculate_event_likelihood(h, 0.2, 0, "B")
        0.16666666666666666
        >>> m.calculate_event_likelihood(h, 0.2, 1, "A")
        0.16666666666666666

        No other change has positive probability.
        >>> for i in "A", "D", "E":
        ...   print(m.calculate_event_likelihood(h, 0.2, 0, "D"))
        0.0
        0.0
        0.0

        2: Copying or mutation
        ~~~~~~~~~~~~~~~~~~~~~~

        Assume the same example as above, but now assume that mutations happen
        with half the rate of anything being copied.

        >>> m.mutation_rate = 0.5

        Then each of the three nodes still has the same probability of being
        the target of a change (1/3), but each copying is twice as likely as a
        random mutation, so if we focus on node 0 as the target of a change: In
        2 out of 5 cases it copies a B from node 1, in 2/5 cases it copies a C
        from node 2, and with probability 1/25 each, it takes a new value of A,
        B, C, D, or E.

        So the probability of a change to A, D, or E is 1/(25*3), whereas a
        change to B or C happens in 2/(5*3) + 1/(25*3) of cases.

        >>> m.languages = range(5)
        >>> m.calculate_event_likelihood(h, 0.2, 0, "B")
        0.14666666666666667
        >>> m.calculate_event_likelihood(h, 0.2, 0, "C")
        0.14666666666666667
        >>> m.calculate_event_likelihood(h, 0.2, 0, "A")
        0.013333333333333332
        >>> m.calculate_event_likelihood(h, 0.2, 0, "D")
        0.013333333333333332
        >>> m.calculate_event_likelihood(h, 0.2, 0, "E")
        0.013333333333333332

        """
        total_rate = (
            self.copy_rate_matrix.sum() + self.mutation_rate * history._states.shape[1]
        )
        p_of_choosing_this_node = (
            self.copy_rate_matrix[:, node].sum() + self.mutation_rate
        ) / total_rate
        odds_state = self.mutation_rate / self.nlang
        total_odds = self.mutation_rate
        for edgefrom, p in enumerate(self.copy_rate_matrix[:, node]):
            neighbor_state = history.before(time, edgefrom)
            if neighbor_state == new_state:
                odds_state += p
            total_odds += p
        return p_of_choosing_this_node * odds_state / total_odds

    def calculate_event_loglikelihood(
        self,
        history: History[L],
        time: float,
        node: int,
        new_state: L,
    ):
        """Calculate the loglikelihood of an event at some point in the history.

        This only calculates the likelihood that at a specific time point, a
        particular node takes the given new value based on earlier events. It
        does not compute the marginal likelihood of that change based on later
        changes. The likelihood is not conditioned on the event being
        observable as an actual change in the system state.

        See also
        ========

        History.calculate_event_likelihood:
            The same likelihood calculation, without the log.
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

        >>> h = History(start, [], end=1.0)
        >>> m = HistoryModel(matrix, 0.0, range(nlang))

        If the first change is at time 0.2, it will have equal probability of
        changing node 0 to B (copied from node 1) or C (copied from node 2);
        changing node 1 to A or C, or changing node 2 to A or B ??? so each has
        probability 1/6.

        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 0, "B"))
        0.16666666666666669
        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 1, "A"))
        0.16666666666666669

        No other change has positive probability.
        >>> for i in "A", "D", "E":
        ...   print(m.calculate_event_loglikelihood(h, 0.2, 0, "D"))
        -inf
        -inf
        -inf

        2: Copying or mutation
        ~~~~~~~~~~~~~~~~~~~~~~

        Assume the same example as above, but now assume that mutations happen
        with half the rate of anything being copied.

        >>> m.mutation_rate = 0.5

        Then each of the three nodes still has the same probability of being
        the target of a change (1/3), but each copying is twice as likely as a
        random mutation, so if we focus on node 0 as the target of a change: In
        2 out of 5 cases it copies a B from node 1, in 2/5 cases it copies a C
        from node 2, and with probability 1/25 each, it takes a new value of A,
        B, C, D, or E.

        So the probability of a change to A, D, or E is 1/(25*3), whereas a
        change to B or C happens in 2/(5*3) + 1/(25*3) of cases.

        >>> m.languages = range(5)
        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 0, "B"))
        0.14666666666666667
        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 0, "C"))
        0.14666666666666667
        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 0, "A"))
        0.013333333333333338
        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 0, "D"))
        0.013333333333333338
        >>> numpy.exp(m.calculate_event_loglikelihood(h, 0.2, 0, "E"))
        0.013333333333333338

        """
        logtotal_rate = numpy.log(
            self.copy_rate_matrix.sum() + self.mutation_rate * history._states.shape[1]
        )
        logp_of_choosing_this_node = (
            numpy.log(self.copy_rate_matrix[:, node].sum() + self.mutation_rate)
            - logtotal_rate
        )
        logodds_state = numpy.log(self.mutation_rate) - numpy.log(self.nlang)
        log_total_odds = numpy.log(self.mutation_rate)
        for edgefrom, p in enumerate(self.copy_rate_matrix[:, node]):
            logp = numpy.log(p)
            neighbor_state = history.before(time, edgefrom)
            if neighbor_state == new_state:
                logodds_state = numpy.logaddexp(logodds_state, logp)
            log_total_odds = numpy.logaddexp(log_total_odds, logp)
        return logp_of_choosing_this_node + logodds_state - log_total_odds

    def calculate_no_change_likelihood(
        self,
        history: History[L],
        time: float,
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
        Mutations happen with half the rate of
        anything being copied.

        >>> h = History(start, [])
        >>> mu = 0.5
        >>> m = HistoryModel(matrix, mu, range(nlang))

        Then out of the 3*2.5 rate-weighted events, the ones that do not change
        the system are

         - Node 0 copies state A from node 1 (at rate 1),
         - Node 0 mutates to state A randomly (p=1/5 at rate 0.5)
        together at likelihood
        >>> m.calculate_event_likelihood(h, 0.2, 0, "A")
        0.14666666666666667

         - Node 1 copies state A from node 0 (at rate 1),
         - Node 1 mutates to state A randomly (p=1/5 at rate 0.5)
        together at likelihood
        >>> m.calculate_event_likelihood(h, 0.2, 1, "A")
        0.14666666666666667

        - Node 2 mutates to state B randomly (p=1/5 at rate 0.5)
        >>> m.calculate_event_likelihood(h, 0.2, 2, "B")
        0.013333333333333332

        Together, this is a likelihood of
        >>> m.calculate_no_change_likelihood(h, 0.2)
        0.30666666666666664

        For two connected nodes with identical values, the only way of change
        is a mutation. If there are only two values, each mutation is
        observable with probability 1/2. If mutation happens at the same rate
        as copying, the probability of an actual change is 1/4.

        >>> HistoryModel(
        ...   numpy.array([[0,1],[1,0]]), 1, ["A", "B"]
        ...   ).calculate_no_change_likelihood(
        ...     History(["A", "A"], []), 0.5)
        0.75

        """
        p = 0
        for node in range(history._states.shape[1]):
            state_before = history.before(time, node)
            p += self.calculate_event_likelihood(
                history,
                time,
                node,
                state_before,
            )
        return p

    def calculate_any_change_loglikelihood(
        self,
        history: History[L],
        time: float,
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
        Mutations happen with half the rate of
        anything being copied.

        >>> h = History(start, [])
        >>> mu = 0.5
        >>> m = HistoryModel(matrix, mu, range(nlang))

        Then out of the 3*2.5 rate-weighted events, the ones that do not change
        the system are

         - Node 0 copies state A from node 1 (at rate 1),
         - Node 0 mutates to state A randomly (p=1/5 at rate 0.5)
        together at likelihood
        >>> numpy.allclose(
        ...   m.calculate_event_likelihood(h, 0.2, 0, "A"),
        ...   (1 + 0.5/5) / (3*2.5))
        True

         - Node 1 copies state A from node 0 (at rate 1),
         - Node 1 mutates to state A randomly (p=1/5 at rate 0.5)
        together at that same likelihood
        >>> m.calculate_event_likelihood(h, 0.2, 1, "A")
        0.14666666666666667

        - Node 2 mutates to state B randomly (p=1/5 at rate 0.5)
        >>> m.calculate_event_likelihood(h, 0.2, 2, "B")
        0.013333333333333332

        Together, this is a likelihood of
        >>> 1-numpy.exp(m.calculate_any_change_loglikelihood(h, 0.2))
        0.30666666666666653

        For two connected nodes with identical values, the only way of change
        is a mutation. If there are only two values, each mutation is
        observable with probability 1/2. If mutation happens at the same rate
        as copying, the probability of an actual change is 1/4.

        >>> l = HistoryModel(
        ...   numpy.array([[0,1],[1,0]]), 1, ["A", "B"]
        ...   ).calculate_any_change_loglikelihood(
        ...     History(["A", "A"], []), 0.5)
        >>> numpy.allclose(numpy.exp(l), 1/4)
        True

        """
        logp = -numpy.inf
        input_rates = self.copy_rate_matrix.sum()
        n_nodes = len(history.start())
        for node in range(n_nodes):
            state_before = history.before(time, node)
            inputs = numpy.arange(n_nodes)[self.copy_rate_matrix[:, node] > 0]
            copy_values = {history.before(time, i) for i in inputs} - {state_before}
            for v in copy_values:
                logp = numpy.logaddexp(
                    logp,
                    self.calculate_event_loglikelihood(
                        history,
                        time,
                        node,
                        v,
                    ),
                )

            # p(change from mutation)
            # = p(this node mutates) * p(it mutates to something not covered yet)
            # = rate(this node mutates) / rate(any event)
            #   * #(potential v's \ v's already covered \ {previous v}) / #(potential v's)
            logp = numpy.logaddexp(
                logp,
                numpy.log(self.mutation_rate)
                - numpy.log(n_nodes * self.mutation_rate + input_rates)
                + numpy.log(self.nlang - len(copy_values) - 1)
                - numpy.log(self.nlang),
            )

        if logp > 0.0:
            return 0.0
        return logp

    def calculate_change_likelihood(
        self,
        history: History[L],
        time: float,
        node: int,
        new_state: L,
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
            if the ???new??? node value is the same as the node value before the change.

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
        >>> m = HistoryModel(matrix, mu, range(nlang))

        Then a first event (let us assume at time 0.2) that sets the new value
        of node 0 to A is no change.

        >>> m.calculate_change_likelihood(h, 0.2, 0, "A")
        Traceback (most recent call last):
          ...
        ValueError: Event is no change

        The only copy events that are observable are those where node 2 copies
        node 0 or node 1 (total rate 2) or vice versa (also total rate 2); in
        addition, 4 out of 5 mutation events for each node are observable, each
        with equal probability (rate 0.1 each). So node 2 gaining state A has a
        probability of (2+0.1)/(2+2+3*0.4) = 21/52
        >>> m.calculate_change_likelihood(h, 0.2, 2, "A") * 52
        20.999999999999996
        >>> m.calculate_change_likelihood(h, 0.2, 2, "C") * 52
        0.9999999999999999

        All the valid likelihoods obviously add up to 1.
        >>> p = 0
        >>> for node in 0,1,2:
        ...   for state in "A", "B", "C", "D", "E":
        ...     try:
        ...       p += m.calculate_change_likelihood(h, 0.2, node, state)
        ...     except ValueError:
        ...       print(node, state)
        ...
        0 A
        1 A
        2 B
        >>> p
        1.0

        """
        if history.before(time, node) == new_state:
            raise ValueError("Event is no change")
        return self.calculate_event_likelihood(history, time, node, new_state) / (
            1 - self.calculate_no_change_likelihood(history, time)
        )

    def calculate_change_loglikelihood(
        self,
        history: History[L],
        time: float,
        node: int,
        new_state: L,
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
            if the ???new??? node value is the same as the node value before the change.

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
        >>> m = HistoryModel(matrix, mu, range(nlang))

        Then a first event (let us assume at time 0.2) that sets the new value
        of node 0 to A is no change.

        >>> m.calculate_change_loglikelihood(h, 0.2, 0, "A")
        Traceback (most recent call last):
          ...
        ValueError: Event is no change

        The only copy events that are observable are those where node 2 copies
        node 0 or node 1 (total rate 2) or vice versa (also total rate 2); in
        addition, 4 out of 5 mutation events for each node are observable, each
        with equal probability (rate 0.1 each). So node 2 gaining state A has a
        probability of (2+0.1)/(2+2+3*0.4) = 21/52
        >>> numpy.allclose(
        ...   m.calculate_change_loglikelihood(h, 0.2, 2, "A") + numpy.log(52),
        ...   numpy.log(21))
        True
        >>> numpy.allclose(
        ...   m.calculate_change_loglikelihood(h, 0.2, 2, "C") + numpy.log(52),
        ...   0)
        True

        All the valid likelihoods obviously add up to 1.
        >>> p = 0
        >>> for node in 0,1,2:
        ...   for state in "A", "B", "C", "D", "E":
        ...     try:
        ...       p += m.calculate_change_likelihood(h, 0.2, node, state)
        ...     except ValueError:
        ...       print(node, state)
        ...
        0 A
        1 A
        2 B
        >>> p
        1.0

        """
        if history.before(time, node) == new_state:
            raise ValueError("Event is no change")
        loglk = self.calculate_event_loglikelihood(history, time, node, new_state)
        logpchange = self.calculate_any_change_loglikelihood(history, time)
        return loglk - logpchange

    def loglikelihood(
        self,
        history,
        start=0.0,
        end=None,
    ):
        """Calculate the complete likelihood of the history.

        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> start = ["A", "A", "B"]
        >>> nlang = 5
        >>> mu = 0.5
        >>> m = HistoryModel(matrix, mu, range(nlang))

        With rates adding up to 7.5, of which 5.2 induce change, the
        probability of observing nothing is rather low (but not as low as taken
        when not conditioning on observable changes).

        The probability of seeing no event within 1 time unit when events have
        a total rate of 5.2 is

        >>> numpy.exp(-5.2)
        0.0055165644207607716

        so that should be (up to some rounding) the likelihood of still seeing
        an empty history after 1 time unit:

        >>> no_changes = History(start, changes=[], end=1.0)
        >>> numpy.allclose(
        ...   numpy.exp(m.loglikelihood(no_changes)),
        ...   numpy.exp(-5.2)
        ... )
        True

        Of course, if all starting values are identical and no mutations
        happen, the likelihood of nothing observable happening any more is 1.0.

        >>> constant = History(["A", "A", "A"], [], 1.0)
        >>> m = HistoryModel(matrix, 0.0, range(100))
        >>> numpy.exp(m.loglikelihood(constant))
        1.0

        If there are events, their likelihood is included. Consider a simple
        2-node model with mutation.

        >>> matrix = numpy.array([[0, 1], [1, 0]])
        >>> mu = 1
        >>> m = HistoryModel(matrix, mu, [0, 1])
        >>> h = History(["A", "A"], [(0.5, 0, "B"), (1.0, 1, "B")], end=2.0)

        The probability to see this history is composed of

         - The probability that the first change happen at t=0.5. There are two
           possible change events, each happening with rate 0.5 (mutation rate
           1 and probability 1/2 of an actual change upon mutation), so this
           probability is exp(-0.5).

         - The probability that ??? given that a change does happen at t=0.5 ???
           node 0 then flips to B. The only other possible change is that node
           1 flips to B, which happens at the same rate, so this has
           probability 1/2.

           >>> m.calculate_change_likelihood(h, 0.5, 0, "B")
           0.5

        >>> numpy.allclose(
        ...   m.loglikelihood(h, end=0.5),
        ...   numpy.log( numpy.exp(-0.5) * 0.5 ))
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
        ...   numpy.exp(m.loglikelihood(h, start=0.5, end=1.0)),
        ...   3*numpy.exp(-1.5) * 0.5)
        True

         - The probability that in the new homogenous state (B, B) no change
           happens for another 1.0 time units, which has probabilty exp(-1).

        >>> numpy.allclose(
        ...   numpy.exp(m.loglikelihood(h, start=1.0)),
        ...   numpy.exp(-1))
        True
        >>> numpy.allclose(
        ...   numpy.exp(m.loglikelihood(h)),
        ...   numpy.exp(-0.5) * 0.5 * 3*numpy.exp(-1.5) * 0.5 * numpy.exp(-1))
        True

        >>> numpy.allclose(
        ...   m.loglikelihood(h, end=0.5) +
        ...   m.loglikelihood(h, start=0.5, end=1.0) +
        ...   m.loglikelihood(h, start=1.0),
        ...   m.loglikelihood(h))
        True

        """
        if end is None:
            end = history.end
        prev_time = start
        logp = 0.0
        state = history._states[0].copy()
        observable_transition = state[None, :] != state[:, None]
        mutation_change_rate = (
            self.mutation_rate
            * history._states.shape[1]
            * (self.nlang - 1)
            / self.nlang
        )
        rate_of_observable_change = (
            self.copy_rate_matrix * observable_transition
        ).sum() + mutation_change_rate
        for time, node, value in history.all_changes():
            if time > end:
                break
            # Contribution from waiting time
            if time > start:
                logp += exponential_distribution.logpdf(
                    (time - prev_time) * rate_of_observable_change
                ) + numpy.log(rate_of_observable_change)
                logp += self.calculate_change_loglikelihood(
                    history,
                    time,
                    node,
                    value,
                )
                prev_time = time

            # Contribution from change
            state[node] = value
            observable_transition[node, :] = state != value
            observable_transition[:, node] = state != value
            rate_of_observable_change = (
                self.copy_rate_matrix * observable_transition
            ).sum() + mutation_change_rate
        return logp + exponential_distribution.logsf(
            (end - prev_time) * rate_of_observable_change
        )

    def alternatives_with_loglikelihood(
        self,
        history: History[L],
        time: float,
        node: int,
    ) -> typing.Iterable[tuple[int, float]]:
        """Calculate the possible alternatives for the specified change.

        In a history where a given node changes at a given time, compute the
        probability of each value the node could take, given the other parts of the
        history.

        """
        alternative_history = deepcopy(history)
        time_slice = bisect.bisect_right(alternative_history._times[:, node], time) - 1
        is_neighbor = (self.copy_rate_matrix[:, node] > 0) | (
            self.copy_rate_matrix[node, :] > 0
        )
        neighbors = numpy.arange(len(is_neighbor))[is_neighbor]
        dependent_changes = list(history.related(time, node, neighbors))
        forbidden_values = set()
        try:
            next_time = history._times[time_slice + 1, node]
            if next_time > history.end:
                raise IndexError
            forbidden_values.add(
                history._states[time_slice + 1, node],
            )
        except IndexError:
            next_time = history.end
        if time_slice > 0:
            forbidden_values.add(history._states[time_slice - 1, node])
        nontrivial_values = (
            {v for _, _, v in dependent_changes}
            # TODO: Technically, only the incoming neighbors are relevant
            # ???before???, not the outgoing neighbors.
            | {history.before(time, neighbor) for neighbor in neighbors}
        ) - forbidden_values
        # Well, that's a lie. The ???trivial??? values are not exactly trivial to
        # compute. But they all have the same likelihood, so they can all be
        # handled the same.
        trivial_values = set(range(self.nlang)) - nontrivial_values - forbidden_values

        try:
            some_trivial_value = [trivial_values.pop()]
        except KeyError:
            some_trivial_value = []
        for value in itertools.chain(nontrivial_values, some_trivial_value):
            likelihood = 1.0
            alternative_history._states[time_slice, node] = value
            loglikelihood = self.loglikelihood(
                alternative_history,
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
                likelihood *= exponential_distribution.pdf(
                    (d_time - time)
                    * (
                        (
                            self.copy_rate_matrix[d_node, node]
                            + self.copy_rate_matrix[node, d_node]
                        )
                    )
                ) * (
                    self.copy_rate_matrix[d_node, node]
                    + self.copy_rate_matrix[node, d_node]
                )
                likelihood *= self.calculate_change_likelihood(
                    alternative_history,
                    d_time,
                    d_node,
                    d_value,
                )
            yield value, loglikelihood
        for value in trivial_values:
            yield value, loglikelihood

    def generate_history(
        self,
        start: typing.Sequence[int],
        end: float,
    ) -> History[int]:
        """Create a new history, according to this model.

        >>> matrix = numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> mu = 0.5
        >>> nlang = 5
        >>> m = HistoryModel(matrix, mu, range(nlang))
        >>> start = ["A", "A", "B"]
        >>> h = m.generate_history(start, 100.)
        >>> m.loglikelihood(h) > -512 or m.loglikelihood(h)
        True

        """
        state = deepcopy(start)

        def changes():
            time = 0
            ps = numpy.cumsum(self.copy_rate_matrix.flat)
            total_rate = ps[-1] + self.mutation_rate * len(state)
            time += exponential_distribution.rvs() * (1 / total_rate)
            while time < end:
                x = numpy.random.random() * total_rate
                if x > ps[-1]:
                    lang = self.languages[numpy.random.randint(self.nlang)]
                    node = numpy.random.randint(len(state))
                    if state[node] != lang:
                        state[node] = lang
                        yield (time, node, state[node])
                else:
                    edgefrom, edgeto = numpy.unravel_index(
                        bisect.bisect(ps, x), self.copy_rate_matrix.shape
                    )
                    if state[edgeto] != state[edgefrom]:
                        state[edgeto] = state[edgefrom]
                        yield (time, edgeto, state[edgeto])
                time += exponential_distribution.rvs() * (1 / total_rate)

        return History(start, changes(), end)


def gibbs_redraw_change(model, history, change=None):
    """A Gibbs operator changing the value of a change.

    A Gibbs operator that changes the value on the interval between two
    subsequent changes of one node, based on all changes on adjacent nodes that
    interact with this change.

    """
    changes = history.all_changes()
    if change is None:
        try:
            time, node, value = changes[numpy.random.randint(len(changes))]
        except ValueError:
            return -numpy.inf, None
    else:
        time, node, value = change
    logsump = -numpy.inf
    logcumprob = []
    logprob = []
    values = []
    logprob_old = -numpy.inf
    for alternative, logprobability in model.alternatives_with_loglikelihood(
        history,
        time,
        node,
    ):
        logprob.append(logprobability)
        logsump = numpy.logaddexp(logsump, logprobability)
        logcumprob.append(logsump)
        values.append(alternative)
        if value == alternative:
            logprob_old = logprobability
    if logsump == -numpy.inf:
        index = numpy.random.randint(len(values))
    else:
        index = bisect.bisect(numpy.exp(logcumprob), numpy.random.random() * logsump)
    new_value = values[index]
    if new_value == value:
        return 0, history
    else:
        return logprob[index] - logprob_old, History(
            history.start(),
            [
                (t, n, v if j != change else new_value)
                for j, (t, n, v) in enumerate(history.all_changes())
            ],
            history.end,
        )


def gibbs_redraw_start(model, history, node=None):
    """A Gibbs operator changing the value of a change.

    A Gibbs operator that changes the value on the interval between two
    subsequent changes of one node, based on all changes on adjacent nodes that
    interact with this change.

    """
    start = list(history.start())
    if node is None:
        node = numpy.random.randint(len(start))
    value = start[node]
    return gibbs_redraw_change(model, history, change=(0.0, node, value))


def move_change(model, history):
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


# TODO:
# Also add a pair of reversible jump operators which split a change into two or merge two adjacent changes into one.


def add_change(model, history):
    pos = numpy.random.uniform(0.0, history.end)
    random_node = numpy.random.randint(len(history.start()))
    after = history.after(pos, random_node)
    # New_value also must differ from the next change, if there is one (so two
    # values are forbidden). Otherwise, subtract only 1.
    if after is None:
        skip = 1
    else:
        skip = 2
    new_value = numpy.random.randint(model.nlang - skip)
    if new_value == history.at(pos, random_node):
        new_value = model.nlang - 1
    if new_value == after:
        new_value = model.nlang - 2
        if new_value == history.at(pos, random_node):
            new_value = model.nlang - 1
    changes = history.all_changes()
    bisect.insort(
        changes,
        (pos, random_node, new_value),
    )
    return -numpy.log(
        history.end * len(history.start()) * (model.nlang - skip)
    ), History(history.start(), changes, history.end)


def removable_change(history: History) -> ():
    """Find a random change that can be removed.

    A change that can be removed is one where the value before the change is
    different from the value after the next change. (Any final change can
    always be removed.)

    Returns
    =======
    change_index, node: int, int
        The 2-part multiindex into history._states and history._times for the change.

    Raises
    ======

    Only a history with no changes has no removable changes. In that case, we
    raise a ValueError.

    >>> h = History(["A", "A"], [], end=2.0)
    >>> removable_change(h)
    Traceback (most recent call last):
    [...]
    ValueError: ("History History(array(['A', 'A'], dtype='<U1'), [], 2.0) has no changes to remove.", History(array(['A', 'A'], dtype='<U1'), [], 2.0))

    Examples
    ========

    If there is only one change, it is deterministically the result of this
    function.

    >>> h = History(["A", "A"], [(0.5, 0, "B")], end=2.0)
    >>> removable_change(h)
    (1, 0)

    In the following example, the first change (which would be described as
    `(1, 0)` like above) is not removable, so the second change is the only
    option.

    >>> h = History(["A", "A"], [(0.5, 0, "B"), (0.6, 0, "A")], end=2.0)
    >>> removable_change(h)
    (2, 0)

    A longer quantitative example:

    >>> from collections import Counter
    >>> from scipy.stats import binomtest
    >>> h = History(["A", "A", "B"], [(0.3, 1, "B"), (0.4, 2, "A"), (0.5, 0, "B"), (0.6, 0, "A"), (0.7, 1, "A"), (0.8, 0, "B"), (0.9, 2, "C")], end=2.0)
    >>> c = Counter()
    >>> for i in range(200):
    ...   c[removable_change(h)] += 1
    >>> c[(3, 0)] + c[(2, 1)] + c[(1, 2)] + c[(2, 2)]
    200
    >>> binomtest(c[(3, 0)], 200, 0.25).pvalue > 1e-3
    True
    >>> binomtest(c[(2, 1)], 200, 0.25).pvalue > 1e-3
    True
    >>> binomtest(c[(1, 2)], 200, 0.25).pvalue > 1e-3
    True
    >>> binomtest(c[(2, 2)], 200, 0.25).pvalue > 1e-3
    True

    """
    valid_changes = numpy.ones(history._times.shape, dtype=bool)
    if len(history._states) > 1:
        valid_changes[1:-1] = history._states[:-2] != history._states[2:]
    n_changes = ((history._times <= history.end) & valid_changes).sum(0) - 1
    heap = n_changes.cumsum()
    if heap[-1] == 0:
        raise ValueError(f"History {history} has no changes to remove.", history)
    random_node = bisect.bisect(heap, numpy.random.randint(heap[-1]))
    random_change = numpy.random.randint(1, n_changes[random_node] + 1)
    for i, v in enumerate(valid_changes[:, random_node]):
        if not v:
            random_change += 1
        elif i >= random_change:
            break
    return random_change, random_node


def remove_change(model, history):
    # Pick a random change that is not undone (i.e. the value before its
    # segment and the value after it are different).
    try:
        (
            random_change,
            random_node,
        ) = removable_change(history)
    except ValueError:
        return -numpy.inf, history

    after = history.after(history._times[random_change, random_node], random_node)
    if after is None:
        skip = 1
    else:
        skip = 2

    changes = history.all_changes()
    changes.remove(
        (
            history._times[random_change, random_node],
            random_node,
            history._states[random_change, random_node],
        )
    )
    return numpy.log(
        history.end * len(history.start()) * (model.nlang - skip)
    ), History(history.start(), changes, history.end)


def add_or_remove_change(model, history):
    if numpy.random.randint(2):
        return add_change(model, history)
    else:
        return remove_change(model, history)


def select_operator(rj=True):
    if rj == "only":
        all_operators = []
    else:
        all_operators = [gibbs_redraw_change, gibbs_redraw_start] + [move_change] * 2
    if rj:
        all_operators.append(add_or_remove_change)
    return all_operators[numpy.random.randint(len(all_operators))]


def step_mcmc(model, history, loglikelihood, rj=True) -> tuple[History, float]:
    """Execute a single MCMC step"""
    operator = select_operator(rj=rj)
    hastings_ratio, candidate_history = operator(model, deepcopy(history))
    if hastings_ratio == -numpy.inf:
        # print(operator.__name__, "rejected: Impossible")
        return history, loglikelihood
    candidate_loglikelihood = model.loglikelihood(candidate_history)
    logp = candidate_loglikelihood + hastings_ratio - loglikelihood
    if (logp > 0) or (numpy.random.random() < numpy.exp(logp)):
        # print(operator.__name__, "accepted")
        history = candidate_history
        # print(history)
    # else: print(operator.__name__, "rejected")

    return history, candidate_loglikelihood


if __name__ == "__main__":
    end = numpy.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2])
    end_time = 100.0
    copy_rate_matrix = numpy.diag([1 for _ in end[1:]], 1) + numpy.diag(
        [1 for _ in end[1:]], -1
    )
    nlang = 200
    history = History(end, [(50, 10, 2)], end=end_time)
    mutation_rate = 1e-4
    model = HistoryModel(copy_rate_matrix, mutation_rate, nlang)
    loglikelihood = model.loglikelihood(history)
    while True:
        history, loglikelihood = step_mcmc(model, history, loglikelihood)
