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

    self.in_indices = [0]
    self.out_indices = []

    self.current_state_value = self.cash_flow_in[0]
    self.best_state_value = self.current_state_value

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

    if self.current_state_value < self.best_state_value \
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


cft = CashFlowTraverser(dummy_set(), dummy_set())
print(cft.try_move([1], [1]))
