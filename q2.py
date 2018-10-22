# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np


#compute all combinations for two portfolios
def question02(cashFlowIn, cashFlowOut):
  # modify and then return the variable below
  state_tensor = LazyStateTensor(cashFlowIn, cashFlowOut)
  answer = state_tensor.find_solution()
  return answer


class LazyStateTensor:
  """
  Defines the problem's state tensor but allows it to be evaluated lazily.
  Code is optimised for performance, not elegance.  State tensors can
  always be assumed to have a positive derivative wrt. index; the value in
  a cell will always increase or stay constant if any of the index
  dimensions are increased.
  """

  def __init__(self, sorted_cash_flow_in, reverse_sorted_cash_flow_out,
      in_subset_size=1, out_subset_size=0):
    """
    [Int] -> [Int] -> Int? -> Int? -> LazyStateTensor
    """
    self.cash_flow_in = sorted_cash_flow_in
    self.cash_flow_out = reverse_sorted_cash_flow_out

    self.in_subset_size = None
    self.out_subset_size = None

    self.rank = None
    self.shape = None

    self.set_subset_sizes(in_subset_size, out_subset_size)

  def set_subset_sizes(self, in_subset_size, out_subset_size):
    """
    Int -> Int -> ()
    Change the subset sizes that the lazy tensor is expecting.
    """
    self.in_subset_size = in_subset_size
    self.out_subset_size = out_subset_size

    self.rank = self.in_subset_size + self.out_subset_size
    self.shape = [len(self.cash_flow_in)] * self.in_subset_size + \
      [len(self.cash_flow_out)] * self.out_subset_size

  def evaluate(self, index):
    """
    [Int] -> Int?
    Evaluate the state tensor at the given index.  If the index is an illegal
    index (it indexes one of the cash flows in the same place twice) then
    None will be returned.
    """
    if not self.is_legal(index):
      return None
    value = 0
    i = 0
    while i < self.in_subset_size:
      value += self.cash_flow_in[index[i]]
      i += 1
    while i < self.rank:
      value -= self.cash_flow_out[index[i]]
      i += 1
    return value

  def is_legal(self, index):
    """
    [Int] -> True
    Determine whether the given index is legal or not.
    """
    seen_values = set()
    i = 0
    while i < self.in_subset_size:
      value = index[i]
      if value in seen_values:
        return False
      seen_values.add(value)
      i += 1
    seen_values = set()
    while i < self.rank:
      value = index[i]
      if value in seen_values:
        return False
      seen_values.add(value)
      i += 1
    return True

  def subtensor_properties(self, index, variable_dimension):
    """
    [Int] -> Int -> (Bool, Int?)
    Given a specific subtensor within the state tensor, determine whether
    or not it only contains non-negative entries and its key entry.  A
    subtensor is said to contain only non-negative entries if evaluating
    every index would yield only values that are >= 0 or illegal.  The
    key entry will be the smallest non-negative value if it contains any,
    otherwise it will be some negative value from the subtensor.  If all
    indices within the subtensor are illegal, the key value will be None.

    If an index is found whose value is zero, it will be returned
    immediately.  In this case, the boolean indicator will be None.
    """
    # TODO: refactor into smaller functions
    subtensor_rank = self.rank - variable_dimension
    if subtensor_rank >= 2:
      iterator = self.subtensor_iterator(index, variable_dimension)
      i = 0
      is_all_non_negative = True
      highest_priority_value = None
      while True:
        current_index = iterator(i)
        if current_index is None:
          break
        current_value = self.evaluate(current_index)
        if current_value is None:
          subtensor_positive, subtensor_highest_priority = \
            self.subtensor_properties(current_index, variable_dimension + 1)
          if subtensor_highest_priority == 0:
            return (None, 0)
          is_all_non_negative = is_all_non_negative and subtensor_positive
          highest_priority_value = self.highest_priority(
            highest_priority_value, subtensor_highest_priority)
        elif current_value < 0:
          _, subtensor_highest_priority = \
            self.subtensor_properties(current_index, variable_dimension + 1)
          if subtensor_highest_priority == 0:
            return (None, 0)
          is_all_non_negative = False
          highest_priority_value = self.highest_priority(
            highest_priority_value, subtensor_highest_priority)
        else:  # Current value is non-negative
          if current_value == 0:
            return (None, 0)
          highest_priority_value = self.highest_priority(
            highest_priority_value, current_value)
          break
        i += 1
      return (is_all_non_negative, highest_priority_value)
    elif subtensor_rank == 1:
      iterator = self.subtensor_iterator(index, variable_dimension)
      i = 0
      is_all_non_negative = True
      highest_priority_value = None
      while True:
        current_index = iterator(i)
        if current_index is None:
          break
        current_value = self.evaluate(current_index)
        if current_value is not None:
          if current_value < 0:
            is_all_non_negative = False
          elif current_value == 0:
            return (None, 0)
          highest_priority_value = self.highest_priority(
            highest_priority_value, current_value)
        i += 1
      return (is_all_non_negative, highest_priority_value)
    else:
      # Rank is 0; should only be reached in edge cases
      value = self.evaluate(index)
      if value > 0 or value is None:
        return (True, value)
      elif value < 0:
        return (False, value)
      else:
        return (None, 0)

  def highest_priority(self, a, b):
    """
    Maybe Int -> Maybe Int -> Maybe Int
    Compare two values, returning the one with the highest priorty.  Non-
    negative integers have the highest priority, with the lowest of two
    non-negative numbers having the greater of the two.  Negative numbers
    have higher priority than None values (illegal indices), both of which
    have lower priority than non-negative integers.
    """
    if self.priority(a) > self.priority(b):
      return a
    else:
      return b

  def priority(self, n):
    """
    Maybe Int -> Int
    Calculate the priority of the value, defined as -1 if the value is
    None, 0 if the value is negative, and 1 / (1 + n) otherwise.
    """
    # TODO: is this really the most efficient definition or priority?
    if n is None:
      return -1
    elif n < 0:
      return 0
    else:
      return 1. / (1 + n)

  def subtensor_iterator(self, index, variable_dimension):
    """
    [Int] -> Int -> (Int -> [Int]?)
    Given a subtensor of the state tensor, specified by its index and the
    index of the dimension to vary, return a function that takes a non-
    negative integer and returns an index within the state tensor, or
    None if the iterator has gone out of bounds.
    """
    max_i = self.shape[variable_dimension] - index[variable_dimension]
    def iterate(i):
      if i >= max_i:
        return None
      index_copy = index[:]
      index_copy[variable_dimension] += i
      return index_copy
    return iterate

  def find_solution(self):
    """
    () -> Int
    Find a solution to the given problem by finding the minimum possible
    value of the state tensor over a range of possible subset size pairs.
    """
    pairs = self.possible_pairs()
    current_best_value = None

    while len(pairs) > 0:
      next_in_size, next_out_size = pairs[0]
      self.set_subset_sizes(next_in_size, next_out_size)

      all_positive, pair_best_value = self.subtensor_properties(
        [0] * self.rank, 0)
      del pairs[0]

      if pair_best_value == 0:
        return 0
      
      current_best_value = self.highest_priority(current_best_value,
        pair_best_value)

      if pair_best_value is not None and pair_best_value < 0:
        # If best value is negative, increasing the number of out values
        # while maintaining the same number of in values will never work
        self.remove_from_list_where(pairs, lambda p: p[0] == next_in_size
          and p[1] > next_out_size)
      if all_positive:
        # If all values are positive, increasing the number of in values
        # while maintaining the same number of out values will never
        # give a better answer than the best found yet
        self.remove_from_list_where(pairs, lambda p: p[0] > next_in_size
          and p[1] == next_out_size)

    return current_best_value

  def possible_pairs(self, order_by=None):
    """
    ((Int, Int) -> Float) -> [(Int, Int)]
    Generate a list of all possible subset size pairs and order
    them according to the given function.
    """
    if order_by is None:
      order_by = lambda p: p[0] + p[1]
    pairs = []
    for i in range(1, len(self.cash_flow_in) + 1):
      for j in range(len(self.cash_flow_out) + 1):
        pairs.append((i, j))
    return sorted(pairs, key=order_by)

  def remove_from_list_where(self, ls, f):
    """
    [a] -> (a -> Bool) -> ()
    Remove elements from a list in-place where the element
    satisfies the given predicate.
    """
    i = len(ls)
    while i > 0:
      i -= 1
      if f(ls[i]):
        del ls[i]


def run_tests():
  assert(question02([66, 293, 215, 188, 147, 326, 449, 162, 46, 350],
    [170, 153, 305, 290, 187]) == 0)
  assert(question02([189, 28], [43, 267, 112, 166]) == 8)
  assert(question02([72, 24, 73, 4, 28, 56, 1, 43], [27]) == 1)
