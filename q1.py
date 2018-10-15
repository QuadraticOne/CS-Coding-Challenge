# ONLY EDIT FUNCTIONS MARKED CLEARLY FOR EDITING

import numpy as np
from random import randint

# modify this function, and create other functions below as you wish
def question01(portfolios):
  if len(portfolios) < 2:
    return None
  # modify and then return the variable below
  answer = maximise_by_culling(portfolios, k=4)
  return answer


def maximise_by_culling(portfolios, k=4):
  """
  [Int] -> Int
  Maximise the XOR of two portfolios in the list of portfolios, first
  by dividing the portfolios up by their first k bits and discarding
  all suboptimal groups, and then maximising pairs in the two
  remaining optimal groups.
  """
  groups = group_by_leading_bits(portfolios, k_max=k)
  group_pairs = find_optimal_group_pairs(groups)
  return max([max([a ^ b for a, b in combine(group_a, group_b)]) \
    for group_a, group_b in group_pairs])


def brute_force(portfolios):
  """
  [Int] -> Int
  Brute force the list of portfolios, comparing the XOR value of each
  pair to find the greatest value.  Works well for n < 1000 but begins
  to get slower around n > 10e5.
  """
  return max(star_map(lambda a, b: a ^ b, portfolios))


def pairs(values, include_duplicates=False):
  """
  [a] -> Bool? -> [(a, a)]
  Find every pair of elements in the given list and return them as a
  list of tuples.  A flag can be set to enable the inclusion of pairs
  consisting of the same element twice.
  """
  outputs = []
  for i in range(len(values)):
    for j in range(i if include_duplicates else i + 1, len(values)):
      outputs.append((values[i], values[j]))
  return outputs


def combine(xs, ys):
  """
  [a] -> [b] -> [(a, b)]
  Combine every element in the first group with every element in the
  second group.
  """
  outputs = []
  for x in xs:
    for y in ys:
      outputs.append((x, y))
  return outputs


def argmax(f, xs):
  """
  (a -> Float) -> [a] -> a
  Return the value from the array that maximises the given function.
  """
  max_element = xs[0]
  max_value = f(max_element)
  for x in xs[1:]:
    test_value = f(x)
    if test_value > max_value:
      max_value = test_value
      max_element = x
  return max_element


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


def find_optimal_group_pairs(leading_bit_groups):
  """
  Dict Int [Int] -> [([Int], [Int])]
  Find the a list of group pairs whose leading bits maximise the XOR function.
  """
  keys = list(leading_bit_groups.keys())
  # Special case where there is only one group:
  if len(keys) < 2:
    key = keys[0]
    return leading_bit_groups[key], leading_bit_groups[key]

  # Calculate the XOR of each leading bits pair
  bit_pair_xors = star_map(lambda a, b: (a, b, a ^ b), keys)
  maximum_xor = argmax(lambda t: t[2], bit_pair_xors)[2]
  optimal_group_pairs = [(a, b) for a, b, xor in bit_pair_xors \
    if xor == maximum_xor \
      and len(leading_bit_groups[a]) > 0 \
      and len(leading_bit_groups[b]) > 0]

  # Extract group contents for each optimal pair
  return [(leading_bit_groups[a], leading_bit_groups[b]) \
    for a, b in optimal_group_pairs]

def group_by_leading_bits(ns, k_max=4):
  """
  [Int] -> Int? -> Dict Int [Int]
  Groups the integers in the list by the value of their first k bits.
  """
  k = min(k_max, most_bits(ns))
  groups = {}
  for n in ns:
    leading_value = leading_bits_value(n, k, 16)
    if leading_value in groups:
      groups[leading_value].append(n)
    else:
      groups[leading_value] = [n]
  return groups


def most_bits(ns):
  """
  [Int] -> Int
  Find the length, in bits, of the largest value in the list.
  """
  return max([len(bin(n)) - 2 for n in ns])


def leading_bits_value(n, k, pad_to_l_bits):
  """
  Int -> Int -> Int
  Pad the binary representation of the given integer to l bits
  if it is less than that length, then extract the value of the
  leading k bits.
  """
  bits = bin(n)[2:]
  bits = ('0' * (pad_to_l_bits - len(bits))) + bits
  return int(bits[:k], 2)


def dummy_portfolios(n):
  """
  Int -> [Int]
  Create a list containing n portfolios sampled from a uniform
  distribution.  Only for testing purposes.
  """
  def random_binary():
    return ''.join([str(randint(0, 1)) for _ in range(16)])
  return [int(random_binary(), 2) for _ in range(n)]
