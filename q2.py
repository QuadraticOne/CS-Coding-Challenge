# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np
from random import randint


MAX_INT = 999999999


#compute all combinations for two portfolios
def question02(cashFlowIn, cashFlowOut):
  # modify and then return the variable below
  cash_flow_in = set(cashFlowIn)
  cash_flow_out = set(cashFlowOut)
  answer = check_for_intersection(cash_flow_in, cash_flow_out)
  return answer


def optimise_ordered_tensor(dereference_function, shape):
  """
  ([Int] -> Int) -> [Int] -> ([Int] -> Int -> Int)
  Produce a function which, when given an index and relevant dimension
  to change, finds the smallest non-negative value in the tensor, where
  the tensor is defined by the given dereference function which exists
  for the specified shape.  Assumes that the dereference function is
  defined such that an increase in any of the indices will result in
  an increase in the return value.
  """
  def find_smallest_non_negative(index, variable_dimension):
    """
    [Int] -> Int -> Int
    Find the smallest non-negative value in the sub-tensor referenced
    by the given index and variable dimension.
    """
    print(index, variable_dimension)
    value_at_index = dereference_function(index)
    if value_at_index >= 0:
      return value_at_index

    if variable_dimension + 1 < len(shape):
      # There are no more values to check (this tensor is a unit)
      return value_at_index if value_at_index >= 0 else MAX_INT
    
    indices_to_check = []
    current_index = index[:]
    smallest_non_negative = MAX_INT
    while current_index[variable_dimension] < shape[variable_dimension] \
      and dereference_function(index) < 0:
      indices_to_check.append(current_index[:])
      current_index[variable_dimension] = current_index[variable_dimension] + 1
    
    value_at_stop_index = dereference_function(index)
    if value_at_stop_index >= 0:
      smallest_non_negative = value_at_stop_index

    return min([optimise_ordered_tensor(i, variable_dimension + 1) \
      for i in indices_to_check] + [smallest_non_negative])

  return find_smallest_non_negative


def state_tensor_evaluator(cash_flow_in, cash_flow_out):
  """
  [Int] -> [Int] -> ([Int] -> [Int] -> Int)
  Return a function that takes a set of indices and outputs the value
  of the state tensor at that position.
  """
  n_out = len(cash_flow_out)
  ci = sorted(list(cash_flow_in))
  co = sorted(list(cash_flow_out))
  def evaluate(in_indices, out_indices):
    """
    [Int] -> [Int] -> Int
    Evaluate the state tensor at the given position.
    """
    return sum([ci[i] for i in in_indices]) - \
      sum([co[n_out - j - 1] for j in out_indices])
  return evaluate


def joint_state_tensor_evaluator(evaluator, m, n):
  """
  ([Int] -> [Int] -> Int) -> Int -> Int -> ([Int]) -> Int
  Take a function which evaluates a state tensor based on its in and
  out indices, and returns a function that does the same job but with
  one index list.
  """
  def split_index_and_evaluate(joint_indices):
    in_indices, out_indices = split_list(joint_indices, m)
    return evaluator(in_indices, out_indices)
  return split_index_and_evaluate


def state_tensor_shape(cash_flow_in, cash_flow_out, m, n):
  """
  [Int] -> [Int] -> Int -> Int -> [Int]
  Find the shape of the state tensor for the given cash flows and
  number of in and out indices.
  """
  return [len(cash_flow_in)] * m + [len(cash_flow_out)] * n


def tabulate_state_tensor(cash_flow_in, cash_flow_out, m, n):
  """
  [Int] -> [Int] -> Int -> Int -> Tensor Int
  Evaluate the entirety of the state tensor for the given m and n,
  where m gives the number of in indices and n gives the number
  of out indices.
  """
  shape = state_tensor_shape(cash_flow_in, cash_flow_out, m, n)
  evaluator = state_tensor_evaluator(cash_flow_in, cash_flow_out)
  def split_index_and_evaluate(joint_indices):
    in_indices, out_indices = split_list(joint_indices, m)
    return evaluator(in_indices, out_indices)
  return iterate_tensor_indices(split_index_and_evaluate, shape)


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


def iterate_tensor_indices(f, shape):
  """
  ([Int] -> a) -> [Int] -> Tensor a
  Iterate over all the indices of a tensor of the given shape,
  mapping the tensor to a function of each index.
  """
  if len(shape) == 0:
    return f([])
  return [iterate_tensor_indices(lambda j: f([i] + j), shape[1:]) \
    for i in range(shape[0])]


def split_list(ls, i):
  """
  [a] -> Int -> ([a], [a])
  Return all elements up until i and all elements after and
  including i in separate lists.
  """
  return ls[:i], ls[i:]
