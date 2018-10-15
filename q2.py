# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np

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
