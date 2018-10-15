# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np
from random import randint

# modify this function, and create other functions below as you wish
def question01(portfolios):
  # modify and then return the variable below
  answer = brute_force(portfolios)
  return answer


def brute_force(portfolios):
  """
  [Int] -> Int
  Brute force the list of portfolios, comparing the XOR value of each
  pair to find the greatest value.  Works well for n < 1000 but begins
  to get slower around n > 10e5.
  """
  return max(star_map(lambda a, b: a ^ b, portfolios))


def star_map(f, values, include_duplicates=False):
  """
  (a -> a -> b) -> [a] -> Bool? -> [b]
  Apply a function to every possible pair of values in the list and
  return the results as a list.  A flag can be set to enable the
  inclusion of pairs consisting of the same element twice.
  """
  outputs = []
  for i in range(len(values)):
    for j in range(i if include_duplicates else i + 1, len(values)):
      outputs.append(f(values[i], values[j]))
  return outputs


def dummy_portfolio(n):
  """
  Int -> [Int]
  Create a list containing n portfolios selected from a uniform
  distribution.  Only for testing purposes.
  """
  def random_binary():
    return ''.join([str(randint(0, 1)) for _ in range(16)])
  return [int(random_binary(), 2) for _ in range(n)]
