import heapq
import math

import numpy as np


class Digits(object):
    """Represents a complete or partial house numbers."""

    def __init__(self, numbers, state, logprob, score):
        self.numbers = numbers
        self.state = state
        self.logprob = logprob
        self.score = score

    def __cmp__(self, other):
        """Compares numbers by score."""
        assert isinstance(other, Digits)
        if self.score == other.score:
            return 0
        elif self.score < other.score:
            return -1
        else:
            return 1
        # For Python 3 compatibility (__cmp__ is deprecated).

    def __lt__(self, other):
        assert isinstance(other, Digits)
        return self.score < other.score
        # Also for Python 3 compatibility.

    def __eq__(self, other):
        assert isinstance(other, Digits)
        return self.score == other.score


class TopN(object):
    """Maintains the top n elements of an incrementally provided set."""

    def __init__(self, n):
        self._n = n
        self._data = []

    def size(self):
        assert self._data is not None
        return len(self._data)

    def push(self, x):
        """Pushes a new element."""
        assert self._data is not None
        if len(self._data) < self._n:
            heapq.heappush(self._data, x)
        else:
            heapq.heappushpop(self._data, x)

    def extract(self, sort=False):
        """Extracts all elements from the TopN. This is a destructive operation.

        The only method that can be called immediately after extract() is reset().

        Args:
          sort: Whether to return the elements in descending sorted order.

        Returns:
          A list of data; the top n elements provided to the set.
        """
        assert self._data is not None
        data = self._data
        self._data = None
        if sort:
            data.sort(reverse=True)
        return data

    def reset(self):
        """Returns the TopN to an empty state."""
        self._data = []


class DigitsInference(object):
    """docstring for DigitsInference"""

    def __init__(self,
                 model,
                 beam_size=3,
                 max_number_length=6):
        super(DigitsInference, self).__init__()
        self.model = model
        self.beam_size = beam_size
        self.max_number_length = max_number_length

        self.start_flag = 10
        self.end_flag = 11

    def beam_search(self, sess, encoded_image):
        initial_state = self.model.feed_image(sess, encoded_image)

        initial_beam = Digits(
            numbers=[self.start_flag],
            state=initial_state[0],
            logprob=0.0,
            score=0.0)

        partial_numbers = TopN(self.beam_size)
        partial_numbers.push(initial_beam)
        complete_numbers = TopN(self.beam_size)
        # Run beam search.
        for _ in range(self.max_number_length - 1):
            partial_numbers_list = partial_numbers.extract()
            partial_numbers.reset()
            input_feed = np.array([c.numbers[-1]
                                   for c in partial_numbers_list])
            state_feed = np.array([c.state for c in partial_numbers_list])
            softmax, new_states = self.model.inference_step(sess,
                                                            input_feed,
                                                            state_feed)
            for i, partial_number in enumerate(partial_numbers_list):
                digit_probabilities = softmax[i]
                state = new_states[i]

                digts_and_probs = list(enumerate(digit_probabilities))
                digts_and_probs.sort(key=lambda x: -x[1])
                digts_and_probs = digts_and_probs[0:self.beam_size]
                for d, p in digts_and_probs:
                    if p < 1e-12:
                        continue  # Avoid log(0).

                    numbers = partial_number.numbers + [d]
                    logprob = partial_number.logprob + math.log(p)
                    score = logprob

                    if d == self.end_flag:
                        beam = Digits(numbers, state, logprob, score)
                        complete_numbers.push(beam)
                    else:
                        beam = Digits(numbers, state, logprob, score)
                        partial_numbers.push(beam)
            if partial_numbers.size() == 0:
                # We have run out of partial candidates; happens when
                # beam_size = 1.
                break
        if not complete_numbers.size():
            complete_numbers = partial_numbers

        return complete_numbers.extract(sort=True)
