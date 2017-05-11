#ifndef KEYVALSORTER_H
#define KEYVALSORTER_H

#include "cpp-utils.h"
#include <stack>

class KeyValSorter
{
  public:
    template <class KeyType, class ValType>
      static void sort(
          std::vector<std::vector<KeyType>> &key,
          std::vector<ValType> &val);

    template <class KeyType, class ValType>
      static void sort(
          size_t left,
          size_t right,
          std::vector<std::vector<KeyType>> &key,
          std::vector<ValType> &val);

  private:
    template <class KeyType, class ValType>
      static void sortAux(
          size_t left,
          size_t right,
          std::vector<std::vector<KeyType>> &key,
          std::vector<ValType> &val);

    template <class KeyType, class ValType>
      static void sortSmall(
          size_t left,
          size_t right,
          std::vector<std::vector<KeyType>> &key,
          std::vector<ValType> &val);
};

// Comparison functions; unrolled up to 4 dimensions
template <class KeyType>
static inline int compKey4(
    size_t i,
    size_t j,
    std::vector<KeyType> &key0,
    std::vector<KeyType> &key1,
    std::vector<KeyType> &key2,
    std::vector<KeyType> &key3)
{
  if (key3[i] < key3[j]) { return -1; }
  else if (key3[i] > key3[j]) { return 1; }
  if (key2[i] < key2[j]) { return -1; }
  else if (key2[i] > key2[j]) { return 1; }
  if (key1[i] < key1[j]) { return -1; }
  else if (key1[i] > key1[j]) { return 1; }
  if (key0[i] < key0[j]) { return -1; }
  else if (key0[i] > key0[j]) { return 1; }
  return 0;
}
template <class KeyType>
static inline int compKey3(
    size_t i,
    size_t j,
    std::vector<KeyType> &key0,
    std::vector<KeyType> &key1,
    std::vector<KeyType> &key2)
{
  if (key2[i] < key2[j]) { return -1; }
  else if (key2[i] > key2[j]) { return 1; }
  if (key1[i] < key1[j]) { return -1; }
  else if (key1[i] > key1[j]) { return 1; }
  if (key0[i] < key0[j]) { return -1; }
  else if (key0[i] > key0[j]) { return 1; }
  return 0;
}

template <class KeyType>
static inline int compKey2(
    size_t i,
    size_t j,
    std::vector<KeyType> &key0,
    std::vector<KeyType> &key1)
{
  if (key1[i] < key1[j]) { return -1; }
  else if (key1[i] > key1[j]) { return 1; }
  if (key0[i] < key0[j]) { return -1; }
  else if (key0[i] > key0[j]) { return 1; }
  return 0;
}

template <class KeyType>
static inline int compKey1(
    size_t i,
    size_t j,
    std::vector<KeyType> &key0)
{
  if (key0[i] < key0[j]) { return -1; }
  else if (key0[i] > key0[j]) { return 1; }
  return 0;
}

template <class KeyType>
static inline int compKey(
    size_t i,
    size_t j,
    std::vector<std::vector<KeyType>> &key)
{
  for (int mode = (int)key.size() - 1; mode >= 0; mode--) {
    KeyType i1 = key[mode][i];
    KeyType i2 = key[mode][j];
    if (i1 < i2) { return -1; }
    else if (i1 > i2) { return 1; }
  }
  return 0;
}

template <class KeyType>
static inline size_t findMedian3(
    size_t i,
    size_t j,
    size_t k,
    std::vector<std::vector<KeyType>> &key)
{
  if (compKey(i, j, key) <= 0) {
    if (compKey(j, k, key) <= 0) { return j; }
    else if (compKey(i, k, key) <= 0) { return k; }
    else { return i; }
  } else {
    if (compKey(i, k, key) <= 0) { return i; }
    else if (compKey(j, k, key) <= 0) { return k; }
    else { return j; }
  }
}

template <class KeyType, class ValType>
static inline void swapElem(
    size_t i,
    size_t j,
    std::vector<std::vector<KeyType>> &key,
    std::vector<ValType> &val)
{
  for (size_t k = 0; k < key.size(); k++) { std::swap(key[k][i], key[k][j]); }
  std::swap(val[i], val[j]);
}

template <class KeyType, class ValType>
static size_t partition(
    size_t left,
    size_t right,
    std::vector<std::vector<KeyType>> &key,
    std::vector<ValType> &val)
{
  if (left == right) { return left; }
  size_t pivot = left, i = left + 1, j = right;
  size_t order = key.size();
  size_t randIdx1 = rand() % (right - left + 1) + left;
  size_t randIdx2 = rand() % (right - left + 1) + left;
  size_t randIdx3 = rand() % (right - left + 1) + left;
  randIdx1 = findMedian3(randIdx1, randIdx2, randIdx3, key);

  // Swap the random pivot to the very left
  swapElem(pivot, randIdx1, key, val);

  /* partition */
  switch(order) {
    case 1:
      while (true) {
        while (i < j && compKey1(i, pivot, key[0]) < 0) { i++; }
        while (compKey1(pivot, j, key[0]) < 0) { j--; }
        if (i >= j) { break; }
        std::swap(key[0][i], key[0][j]);
        std::swap(val[i], val[j]);
        i++;
        j--;
      }
      break;
    case 2:
      while (true) {
        while (i < j && compKey2(i, pivot, key[0], key[1]) < 0) { i++; }
        while (compKey2(pivot, j, key[0], key[1]) < 0) { j--; }
        if (i >= j) { break; }
        std::swap(key[0][i], key[0][j]);
        std::swap(key[1][i], key[1][j]);
        std::swap(val[i], val[j]);
        i++;
        j--;
      }
      break;
    case 3:
      while (true) {
        while (i < j && compKey3(i, pivot, key[0], key[1], key[2]) < 0) { i++; }
        while (compKey3(pivot, j, key[0], key[1], key[2]) < 0) { j--; }
        if (i >= j) { break; }
        std::swap(key[0][i], key[0][j]);
        std::swap(key[1][i], key[1][j]);
        std::swap(key[2][i], key[2][j]);
        std::swap(val[i], val[j]);
        i++;
        j--;
      }
      break;
    case 4:
      while (true) {
        while (i < j && compKey4(i, pivot, key[0], key[1], key[2], key[3]) < 0) { i++; }
        while (compKey4(pivot, j, key[0], key[1], key[2], key[3]) < 0) { j--; }
        if (i >= j) { break; }
        std::swap(key[0][i], key[0][j]);
        std::swap(key[1][i], key[1][j]);
        std::swap(key[2][i], key[2][j]);
        std::swap(key[3][i], key[3][j]);
        std::swap(val[i], val[j]);
        i++;
        j--;
      }
      break;
    default:
      while (true) {
        while (i < j && compKey(i, pivot, key) < 0) { i++; }
        while (compKey(pivot, j, key) < 0) { j--; }
        if (i >= j) { break; }
        for (size_t k = 0; k < key.size(); k++) { std::swap(key[k][i], key[k][j]); }
        std::swap(val[i], val[j]);
        i++;
        j--;
      }
      break;
  }
  // Swap the pivot to the middle
  swapElem(pivot, j, key, val);

  return j;
}

template <class KeyType, class ValType>
void KeyValSorter::sortSmall(
    size_t left,
    size_t right,
    std::vector<std::vector<KeyType>> &key,
    std::vector<ValType> &val)
{
  // Insertion sort
  for (size_t i = left; i < right; i++) {
    // Find the minimum element's key
    size_t minIdx = i;
    for (size_t j = i + 1; j <= right; j++) {
      if (compKey(j, minIdx, key) == -1) { minIdx = j; }
    }
    // Swap to the current position
    for (size_t k = 0; k < key.size(); k++) { std::swap(key[k][minIdx], key[k][i]); }
    std::swap(val[minIdx], val[i]);
  }
}

template <class KeyType, class ValType>
void KeyValSorter::sortAux(
    size_t left,
    size_t right,
    std::vector<std::vector<KeyType>> &key,
    std::vector<ValType> &val)
{
  std::stack<std::pair<size_t, size_t>> sortStack;   
  sortStack.push(std::pair<size_t, size_t>(left, right));

  while (sortStack.empty() == false) {
    left = sortStack.top().first;
    right = sortStack.top().second;
    sortStack.pop();

    if (right - left < 60) { KeyValSorter::sortSmall(left, right, key, val); continue; }

    size_t j = partition(left, right, key, val);

    if (left + 1 < j) { sortStack.push(std::pair<size_t, size_t>(left, j - 1)); }
    if (j + 1 < right) { sortStack.push(std::pair<size_t, size_t>(j + 1, right)); }
  }
}

template <class KeyType, class ValType>
void KeyValSorter::sort(
    size_t left,
    size_t right,
    std::vector<std::vector<KeyType>> &key,
    std::vector<ValType> &val)
{
  sortAux(left, right, key, val);
}

template <class KeyType, class ValType>
void KeyValSorter::sort(
    std::vector<std::vector<KeyType>> &key,
    std::vector<ValType> &val)
{
  if (key[0].size() > 0) { sort((size_t)0, key[0].size() - 1, key, val); }
}

#endif
