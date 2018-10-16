# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np
from random import randint

#compute all combinations for two portfolios
def question02(cashFlowIn, cashFlowOut):
  # modify and then return the variable below
  cash_flow_in = set(cashFlowIn)
  cash_flow_out = set(cashFlowOut)
  answer = check_for_intersection(cash_flow_in, cash_flow_out)
  return answer


def take_smallest_value(in_set, _):
  """
  Set Int -> Set Int -> Int
  A naive but simple solution which takes the smallest element from
  the cash flow in set and returns it.
  """
  return min(in_set)


def check_for_intersection(in_set, out_set):
  """
  Set Int -> Set Int -> Int
  A naive but simple solution which checks whether there are any
  elements common to both sets, and returns 0 if there are.  If not,
  returns the smallest element from the in set.
  """
  if len(in_set.intersection(out_set)) > 0:
    return 0
  return take_smallest_value(in_set, out_set)


def dummy_set(max_length=1000, max_transaction_size=100):
  """
  Int? -> Int? -> [Int]
  Create dummy data with an optionally specified maximum length
  and maximum transaction size.
  """
  length = randint(1, max_length)
  return [randint(1, max_transaction_size) for _ in range(length)]


class CashFlowController:
  """
  Contains algorithms that control a CashFlowTraverser in order to
  try to get an optimal outcome.
  """

  def __init__(self, traverser):
    """
    CashFlowTraverser -> CashFlowController
    """
    self.traverser = traverser

  def initial_traversal(self):
    """
    Perform the initial traversal, initially with 1 index for the ins
    and then with 1 for ins and 1 for outs.
    """
    move_was_valid, _ = self.traverser.try_move([0], [])
    move_was_valid, _ = self.traverser.try_move([0], [0])
    while move_was_valid:
      if self.traverser.current_state_value >= 0:
        move_was_valid, _ = self.increment_out_index(0)
      else:
        move_was_valid, _ = self.increment_in_index(0)
    return self.traverser.best_state_value

  def increment_in_index(self, index_of_index):
    """
    Int -> (Bool, Int?)
    Try to increment the index to the cash flow in indexed by the
    given value, returning the result when the move is tried.
    """
    current_index = self.traverser.in_indices[index_of_index]
    incremented = self.traverser.next_available_in_index(current_index)
    if incremented is None:
      return (False, None)
    new_indices = self.traverser.in_indices[:]
    new_indices[index_of_index] = incremented
    return self.traverser.try_move(new_indices, self.traverser.out_indices)

  def increment_out_index(self, index_of_index):
    """
    Int -> (Bool, Int?)
    Try to increment the index to the cash flow out indexed by the
    given value, returning the result when the move is tried.
    """
    current_index = self.traverser.out_indices[index_of_index]
    incremented = self.traverser.next_available_out_index(current_index)
    if incremented is None:
      return (False, None)
    new_indices = self.traverser.out_indices[:]
    new_indices[index_of_index] = incremented
    return self.traverser.try_move(self.traverser.in_indices, new_indices)


class CashFlowTraverser:
  """
  Attempts to find the optimal solution to the problem by traversing
  the sets in a structured way.
  """

  def __init__(self, cash_flow_in, cash_flow_out):
    """
    [Int] -> [Int] -> CashFlowController
    """
    self.cash_flow_in = sorted(list(cash_flow_in))
    self.cash_flow_out = sorted(list(cash_flow_out))

    self.n_in = len(cash_flow_in)
    self.n_out = len(cash_flow_out)

    self.in_indices = None
    self.out_indices = None

    self.current_state_value = None
    self.best_state_value = -1

  def try_move(self, in_indices, out_indices):
    """
    [Int] -> [Int] -> (Bool, Int?)
    Try to move to the given state.  Returns True if the move was valid
    and was executed.  If the integer component of the return value is
    not None, there are no more possible moves and the best state
    obtained will has been given.
    """
    if len(in_indices) > self.n_in or len(out_indices) > self.n_out:
      return (False, self.best_state_value)
    
    if self.is_state_valid(in_indices, out_indices):
      self.in_indices = in_indices
      self.out_indices = out_indices
      self.set_state_value()
      if self.best_state_value == 0:
        return (True, 0)
      else:
        return (True, None)
    else:
      return (False, None)

  def set_state_value(self):
    """
    () -> ()
    Calculates the value of the current state, and updates the best
    state value if necessary.
    """
    self.current_state_value = \
      sum([self.cash_flow_in[i] for i in self.in_indices]) - \
      sum([self.cash_flow_out[i] for i in self.out_indices])

    if (self.current_state_value < self.best_state_value or
        self.best_state_value < 0) \
      and self.current_state_value >= 0:
      self.best_state_value = self.current_state_value

  def is_state_valid(self, in_indices, out_indices):
    """
    [Int] -> [Int] -> Bool
    Determine whether or not the given state is valid.
    """
    return \
      all([0 <= i < self.n_in for i in in_indices]) and \
      all([0 <= i < self.n_out for i in out_indices]) and \
      all_unique(in_indices) and all_unique(out_indices)

  def next_available_in_index(self, i):
    """
    Int -> Int?
    Return the next available index in the cash flow in that is either
    equal to or greater than the given index.  Returns None if there
    are no available indices.
    """
    if i >= self.n_in:
      return None
    elif i not in self.in_indices:
      return i
    else:
      return self.next_available_in_index(i + 1)

  def next_available_out_index(self, i):
    """
    Int -> Int?
    Return the next available index in the cash flow out that is either
    equal to or greater than the given index.  Returns None if there
    are no available indices.
    """
    if i >= self.n_out:
      return None
    elif i not in self.out_indices:
      return i
    else:
      return self.next_available_out_index(i + 1)


def all_unique(ls):
  """
  [a] -> Bool
  Determine whether the list contains only unique elements.
  """
  return len(ls) == len(set(ls))


# Testing
cft = CashFlowTraverser(dummy_set(10), dummy_set(8))
print(cft.cash_flow_in, cft.cash_flow_out)
controller = CashFlowController(cft)
print(controller.initial_traversal())
