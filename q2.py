# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np


#compute all combinations for two portfolios
def question02(cashFlowIn, cashFlowOut):
  # modify and then return the variable below
  answer = -1
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

    self.set_subset_sizes(in_subset_size, out_subset_size)

  def set_subset_sizes(self, in_subset_size, out_subset_size):
    """
    Int -> Int -> ()
    Change the subset sizes that the lazy tensor is expecting.
    """
    self.in_subset_size = in_subset_size
    self.out_subset_size = out_subset_size

    self.rank = self.in_subset_size + self.out_subset_size

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


def run_tests():
  assert(question02([66, 293, 215, 188, 147, 326, 449, 162, 46, 350],
    [170, 153, 305, 290, 187]) == 0)
  assert(question02([189, 28], [43, 267, 112, 166], 2, 2) == 8)
  assert(question02([72, 24, 73, 4, 28, 56, 1, 43], [27]) == 1)


lst = LazyStateTensor(sorted([66, 293, 215, 188, 147, 326, 449, 162, 46, 350]),
  sorted([170, 153, 305, 290, 187])[::-1], 2, 1)
print(lst.evaluate([1, 1, 0]))
