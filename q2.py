# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np
from random import randint


MAX_INT = 999999999


#compute all combinations for two portfolios
def question02(cashFlowIn, cashFlowOut):
  # modify and then return the variable below
  answer = solve_by_state_tensor(cash_flow_in, cash_flow_out, 2, 2)
  return answer


class ProblemParameters:
  """
  Stores the parameters relating to the problem in one place so they can
  be passed around easily.
  """

  def __init__(self, unsorted_cash_flow_in, unsorted_cash_flow_out,
      in_set_size, out_set_size):
    """
    [Int] -> [Int] -> Int -> Int -> ProblemParameters
    """
    self.unsorted_cash_flow_in = unsorted_cash_flow_in
    self.unsorted_cash_flow_out = unsorted_cash_flow_out

    self.cash_flow_in = sorted(self.unsorted_cash_flow_in)
    self.cash_flow_out = sorted(self.unsorted_cash_flow_out)

    self.in_flow_size = len(self.cash_flow_in)
    self.out_flow_size = len(self.cash_flow_out)

    self.in_set_size = in_set_size
    self.out_set_size = out_set_size

    self.state_tensor_shape = state_tensor_shape(self)
    self.state_tensor_rank = len(self.state_tensor_shape)

    self.unjoined_value_at = state_tensor_evaluator(self)
    self.value_at = joint_state_tensor_evaluator(self)

  def change_set_sizes(self, new_in_set_size=None, new_out_set_size=None):
    """
    Int? -> Int? -> ProblemParameters
    Create a new instance of the problem with different set sizes.
    """
    return ProblemParameters(self.unsorted_cash_flow_in,
      self.unsorted_cash_flow_out, new_in_set_size or self.in_set_size,
      new_out_set_size or self.out_set_size)


def solve_by_state_tensor(params):
  """
  ProblemParameters -> Int
  Solve the problem for two specific cash flows by specifying the size
  of set that will be taken from each cash flow.
  """
  return find_smallest_non_negative(params, [0] * params.state_tensor_rank, 0)


def find_smallest_non_negative(params, index, variable_dimension):
  """
  ProblemParameters -> [Int] -> Int -> Int
  Given a ProblemParameters object describing a state tensor, find the
  smallest non-negative value in the sub-tensor referenced by the given
  index.  This function assumes that the derivative of value with
  respect to the index is always non-negative regardless of the
  dimension being incremented.  Return MAX_INT if there are no non-
  negative values in the sub-tensor.
  """
  index_value = params.value_at(index)
  if variable_dimension >= params.state_tensor_rank or index_value >= 0:
    return index_value if index_value >= 0 else MAX_INT
  negative_indices, first_positive = negatives_and_first_positive(
    params, index, variable_dimension)
  return min([find_smallest_non_negative(params, i, variable_dimension + 1) \
    for i in negative_indices] + [first_positive])


def negatives_and_first_positive(params, index, variable_dimension):
  """
  ProblemParameters -> [Int] -> Int -> ([[Int], Int])
  By varying the given dimension of an index to a state tensor, find
  all the negative elements until a positive occurs and return their
  indices along with the positive value.  If no positive values are
  found then MAX_INT will be returned for the positive value instead.
  """
  iterator = iterated_dimension(params, index, variable_dimension)
  negatives = []
  i = 0
  while True:
    index, value = iterator(i)
    if value is None:
      break
    elif value < 0:
      negatives.append(index)
      i += 1
    else:
      return (negatives, value)
  return (negatives, MAX_INT)


def iterated_dimension(params, index, variable_dimension):
  """
  ProblemParameters -> [Int] -> Int -> (Int -> ([Int]?, Int?))
  Return a function which, given an input integer, returns an index
  within the state tensor along with its state value, or None if the
  dimension's limit has been exceeded.  The index of this new function
  is zero-based.  The indices included are obtained by increasing
  the specified dimension of the tensor, starting from the index
  given.
  """
  out_of_bounds_index = params.state_tensor_shape[variable_dimension] - \
    index[variable_dimension]
  def dimension_at(i):
    """
    Int -> ([Int], Int)?
    """
    if i >= out_of_bounds_index:
      return (None, None)
    else:
      current_index = index[:]
      current_index[variable_dimension] = index[variable_dimension] + i
      return (current_index, params.value_at(current_index))
  return dimension_at


def state_tensor_evaluator(params):
  """
  ProblemParameters -> ([Int] -> [Int] -> Int)
  Return a function that takes a set of indices and outputs the value
  of the state tensor at that position.  If the index is illegal, returns
  a very large negative number.
  """
  def all_unique(ls):
    return len(set(ls)) == len(ls)

  def evaluate(in_indices, out_indices):
    """
    [Int] -> [Int] -> Int
    Evaluate the state tensor at the given position.
    """
    if (not all_unique(in_indices)) or (not all_unique(out_indices)):
      return -MAX_INT

    return sum([params.cash_flow_in[i] for i in in_indices]) - \
      sum([params.cash_flow_out[params.out_flow_size - j - 1] \
      for j in out_indices])
  return evaluate


def joint_state_tensor_evaluator(params):
  """
  ProblemParameters -> ([Int]) -> Int
  Take a function which evaluates a state tensor based on its in and
  out indices, and returns a function that does the same job but with
  one index list.
  """
  def split_index_and_evaluate(joint_indices):
    in_indices, out_indices = split_list(joint_indices, params.in_set_size)
    return params.unjoined_value_at(in_indices, out_indices)
  return split_index_and_evaluate


def state_tensor_shape(params):
  """
  ProblemParameters -> [Int]
  Find the shape of the state tensor for the given cash flows and
  number of in and out indices.
  """
  return [len(params.cash_flow_in)] * params.in_set_size + \
    [len(params.cash_flow_out)] * params.out_set_size


def tabulate_state_tensor(params):
  """
  ProblemParameters -> Tensor Int
  Evaluate the entirety of the state tensor for the given m and n,
  where m gives the number of in indices and n gives the number
  of out indices.
  """
  return iterate_tensor_indices(params.value_at,
    params.state_tensor_shape)


def check_for_intersection(params):
  """
  ProblemParameters -> Int
  A naive but simple solution which checks whether there are any
  elements common to both sets, and returns 0 if there are.  If not,
  returns the smallest element from the in set.
  """
  in_set = set(params.cash_flow_in)
  out_set = set(params.cash_flow_out)
  if len(in_set.intersection(out_set)) > 0:
    return 0
  return params.cash_flow_in[0]


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


# Testing
ps = ProblemParameters(dummy_set(20), dummy_set(20), 2, 2)
print(solve_by_state_tensor(ps))
