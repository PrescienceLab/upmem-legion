using namespace Realm;

////////////////////////////////////////////////////////////////////////
//
// class Point<N,T>

template <int N, typename T>
inline Point<N, T>::Point(value_type val)
{
  for(int i = 0; i < N; i++)
    values[i] = val;
}

template <int N, typename T>
template <typename Arg0, typename Arg1, typename... Args>
inline Point<N, T>::Point(Arg0 val0, Arg1 val1, Args... vals)
  // TODO(cperry): Very bad!  We should not static_cast these for the user, as it can
  // hide sign and casting issues.  Kept here for compatibility while we weed out all
  // the cases in our codebase
  : values{static_cast<value_type>(val0), static_cast<value_type>(val1),
           static_cast<value_type>(vals)...}
{}

template <int N, typename T>
template <typename T2>

inline Point<N, T>::Point(T2 val, T2)
{
  for(int i = 0; i < N; i++)
    values[i] = val;
}

template <int N, typename T>
template <typename T2>

inline Point<N, T>::Point(T2 vals[N], T2)
{
  for(int i = 0; i < N; i++)
    values[i] = vals[i];
}

template <int N, typename T>
template <typename T2>
inline Point<N, T>::Point(const Point<N, T2> &copy_from)
{
  for(int i = 0; i < N; i++)
    values[i] = copy_from.values[i];
}

template <int N, typename T>
template <typename T2>
inline Point<N, T> &Point<N, T>::operator=(const Point<N, T2> &copy_from)
{
  for(int i = 0; i < N; i++)
    values[i] = copy_from.values[i];
  return *this;
}

template <int N, typename T>
inline T &Point<N, T>::operator[](int index)
{
  assert(index < N);
  return values[index];
}

template <int N, typename T>
inline const T &Point<N, T>::operator[](int index) const
{
  assert(index < N);
  return values[index];
}

template <int N, typename T>
template <typename T2>
inline T Point<N, T>::dot(const Point<N, T2> &rhs) const
{
  T acc = values[0] * rhs.values[0];
  for(int i = 1; i < N; i++)
    acc += values[i] * rhs.values[i];
  return acc;
}

template <int N, typename T>
T &Point<N, T>::x()
{
  static_assert(N > 0, "fix ur dimensions");
  return values[0];
}
template <int N, typename T>
T &Point<N, T>::y()
{
  static_assert(N > 1, "fix ur dimensions");
  return values[1];
}
template <int N, typename T>
T &Point<N, T>::z()
{
  static_assert(N > 2, "fix ur dimensions");
  return values[2];
}
template <int N, typename T>
T &Point<N, T>::w()
{
  static_assert(N > 3, "fix ur dimensions");
  return values[3];
}

template <int N, typename T>
const T &Point<N, T>::x() const
{
  static_assert(N > 0, "fix ur dimensions");
  return values[0];
}
template <int N, typename T>
const T &Point<N, T>::y() const
{
  static_assert(N > 1, "fix ur dimensions");
  return values[1];
}
template <int N, typename T>
const T &Point<N, T>::z() const
{
  static_assert(N > 2, "fix ur dimensions");
  return values[2];
}
template <int N, typename T>
const T &Point<N, T>::w() const
{
  static_assert(N > 3, "fix ur dimensions");
  return values[3];
}

template <int N, typename T>
/*static*/ constexpr inline Point<N, T> Point<N, T>::ZEROES(void)
{
  return Point<N, T>(static_cast<T>(0));
}

template <int N, typename T>
/*static*/ constexpr inline Point<N, T> Point<N, T>::ONES(void)
{
  return Point<N, T>(static_cast<T>(1));
}

// component-wise operators defined on Point<N,T> (with optional coercion)
template <int N, typename T, typename T2>
inline bool operator==(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    if(lhs[i] != rhs[i])
      return false;
  return true;
}

template <int N, typename T, typename T2>
inline bool operator!=(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    if(lhs[i] != rhs[i])
      return true;
  return false;
}

template <int N, typename T, typename T2>
inline Point<N, T> operator+(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  Point<N, T> out;
  for(int i = 0; i < N; i++)
    out[i] = lhs[i] + rhs[i];
  return out;
}

template <int N, typename T, typename T2>
inline Point<N, T> &operator+=(Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    lhs[i] += rhs[i];
  return lhs;
}

template <int N, typename T, typename T2>
inline Point<N, T> operator-(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  Point<N, T> out;
  for(int i = 0; i < N; i++)
    out[i] = lhs[i] - rhs[i];
  return out;
}

template <int N, typename T, typename T2>
inline Point<N, T> &operator-=(Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    lhs[i] -= rhs[i];
  return lhs;
}

template <int N, typename T, typename T2>
inline Point<N, T> operator*(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  Point<N, T> out;
  for(int i = 0; i < N; i++)
    out[i] = lhs[i] * rhs[i];
  return out;
}

template <int N, typename T, typename T2>
inline Point<N, T> &operator*=(Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    lhs[i] *= rhs[i];
  return lhs;
}

template <int N, typename T, typename T2>
inline Point<N, T> operator/(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  Point<N, T> out;
  for(int i = 0; i < N; i++)
    out[i] = lhs[i] / rhs[i];
  return out;
}

template <int N, typename T, typename T2>
inline Point<N, T> &operator/=(Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    lhs[i] /= rhs[i];
  return lhs;
}

template <int N, typename T, typename T2>
inline Point<N, T> operator%(const Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  Point<N, T> out;
  for(int i = 0; i < N; i++)
    out[i] = lhs[i] % rhs[i];
  return out;
}

template <int N, typename T, typename T2>
inline Point<N, T> &operator%=(Point<N, T> &lhs, const Point<N, T2> &rhs)
{
  for(int i = 0; i < N; i++)
    lhs[i] %= rhs[i];
  return lhs;
}

////////////////////////////////////////////////////////////////////////
//
// class Rect<N,T>

template <int N, typename T>
inline Rect<N, T>::Rect(void)
{}

template <int N, typename T>
inline Rect<N, T>::Rect(const Point<N, T> &_lo, const Point<N, T> &_hi)
  : lo(_lo)
  , hi(_hi)
{}

template <int N, typename T>
template <typename T2>
inline Rect<N, T>::Rect(const Rect<N, T2> &copy_from)
  : lo(copy_from.lo)
  , hi(copy_from.hi)
{}

template <int N, typename T>
template <typename T2>
inline Rect<N, T> &Rect<N, T>::operator=(const Rect<N, T2> &copy_from)
{
  lo = copy_from.lo;
  hi = copy_from.hi;
  return *this;
}

template <int N, typename T>
inline /*static*/ Rect<N, T> Rect<N, T>::make_empty(void)
{
  Rect<N, T> r;
  T v = T(); // assume any user-defined default constructor initializes things
  for(int i = 0; i < N; i++)
    r.hi[i] = v;
  ++v;
  for(int i = 0; i < N; i++)
    r.lo[i] = v;
  return r;
}

template <int N, typename T>
inline bool Rect<N, T>::empty(void) const
{
  for(int i = 0; i < N; i++)
    if(lo[i] > hi[i])
      return true;
  return false;
}

template <int N, typename T>
inline size_t Rect<N, T>::volume(void) const
{
  size_t v = 1;
  for(int i = 0; i < N; i++)
    if(lo[i] > hi[i])
      return 0;
    else {
      // have to convert both 'hi' and 'lo' to size_t before subtracting
      //  to avoid potential signed integer overflow
      v *= (static_cast<size_t>(hi[i]) - static_cast<size_t>(lo[i]) + 1);
    }
  return v;
}

template <int N, typename T>
inline bool Rect<N, T>::contains(const Point<N, T> &p) const
{
  for(int i = 0; i < N; i++)
    if((p[i] < lo[i]) || (p[i] > hi[i]))
      return false;
  return true;
}

// true if all points in other are in this rectangle
template <int N, typename T>
inline bool Rect<N, T>::contains(const Rect<N, T> &other) const
{
  // containment is weird w.r.t. emptiness: if other is empty, the answer is
  //  always true - if we're empty, the answer is false, unless other was empty
  // this means we can early-out with true if other is empty, but have to remember
  //  our emptiness separately
  bool ctns = true;
  for(int i = 0; i < N; i++) {
    if(other.lo[i] > other.hi[i])
      return true; // other is empty
    // now that we know the other is nonempty, we need:
    //  lo[i] <= other.lo[i] ^ other.hi[i] <= hi[i]
    // (which can only be satisfied if we're non-empty)
    if((lo[i] > other.lo[i]) || (other.hi[i] > hi[i]))
      ctns = false;
  }
  return ctns;
}

// true if there are any points in the intersection of the two rectangles
template <int N, typename T>
inline bool Rect<N, T>::overlaps(const Rect<N, T> &other) const
{
  // overlapping requires there be an element that lies in both ranges, which
  //  is equivalent to saying that both lo's are <= both hi's - this catches
  //  cases where either rectangle is empty
  for(int i = 0; i < N; i++)
    if((lo[i] > hi[i]) || (lo[i] > other.hi[i]) || (other.lo[i] > hi[i]) ||
       (other.lo[i] > other.hi[i]))
      return false;
  return true;
}

template <int N, typename T>
inline Rect<N, T> Rect<N, T>::intersection(const Rect<N, T> &other) const
{
  Rect<N, T> out;
  for(int i = 0; i < N; i++) {
    out.lo[i] = (lo[i] < other.lo[i]) ? other.lo[i] : lo[i]; // max
    out.hi[i] = (hi[i] < other.hi[i]) ? hi[i] : other.hi[i]; // min
  }
  return out;
};

template <int N, typename T>
inline Rect<N, T> Rect<N, T>::union_bbox(const Rect<N, T> &other) const
{
  if(empty())
    return other;
  if(other.empty())
    return *this;
  // the code below only works if both rectangles are non-empty
  Rect<N, T> out;
  for(int i = 0; i < N; i++) {
    out.lo[i] = (lo[i] < other.lo[i]) ? lo[i] : other.lo[i]; // min
    out.hi[i] = (hi[i] < other.hi[i]) ? other.hi[i] : hi[i]; // max
  }
  return out;
};

template <int N, typename T>
template <int N2, typename T2>
inline Rect<N2, T2> Rect<N, T>::apply_transform(const Matrix<N2, N, T2> &transform,
                                                const Point<N2, T2> &offset) const
{
  Rect<N2, T2> tranformed_rect(offset, offset);
  for(int i = 0; i < N2; i++) {
    for(int j = 0; j < N; j++) {
      T2 e = transform.rows[i][j];
      if(e > 0) {
        tranformed_rect.lo[i] += e * lo[j];
        tranformed_rect.hi[i] += e * hi[j];
      }
      if(e < 0) {
        tranformed_rect.lo[i] += e * hi[j];
        tranformed_rect.hi[i] += e * lo[j];
      }
    }
  }
  return tranformed_rect;
}

template <int N, typename T, typename T2>
inline bool operator==(const Rect<N, T> &lhs, const Rect<N, T2> &rhs)
{
  return (lhs.lo == rhs.lo) && (lhs.hi == rhs.hi);
}

template <int N, typename T, typename T2>
inline bool operator!=(const Rect<N, T> &lhs, const Rect<N, T2> &rhs)
{
  return (lhs.lo != rhs.lo) || (lhs.hi != rhs.hi);
}

// rectangles may be displaced by a vector (i.e. point)
template <int N, typename T, typename T2>
inline Rect<N, T> operator+(const Rect<N, T> &lhs, const Point<N, T2> &rhs)
{
  return Rect<N, T>(lhs.lo + rhs, lhs.hi + rhs);
}

template <int N, typename T, typename T2>
inline Rect<N, T> &operator+=(Rect<N, T> &lhs, const Point<N, T2> &rhs)
{
  lhs.lo += rhs;
  lhs.hi += rhs;
  return lhs;
}

template <int N, typename T, typename T2>
inline Rect<N, T> operator-(const Rect<N, T> &lhs, const Point<N, T2> &rhs)
{
  return Rect<N, T>(lhs.lo - rhs, lhs.hi - rhs);
}

template <int N, typename T, typename T2>
inline Rect<N, T> &operator-=(Rect<N, T> &lhs, const Rect<N, T2> &rhs)
{
  lhs.lo -= rhs;
  lhs.hi -= rhs;
  return lhs;
}

////////////////////////////////////////////////////////////////////////
//
// class Matrix<M,N,T>

template <int M, int N, typename T>
inline Matrix<M, N, T>::Matrix(void)
{}

// copies allow type coercion (assuming the underlying type does)
template <int M, int N, typename T>
template <typename T2>
inline Matrix<M, N, T>::Matrix(const Matrix<M, N, T2> &copy_from)
{
  for(int i = 0; i < M; i++)
    rows[i] = copy_from[i];
}

template <int M, int N, typename T>
template <typename T2>
inline Matrix<M, N, T> &Matrix<M, N, T>::operator=(const Matrix<M, N, T2> &copy_from)
{
  for(int i = 0; i < M; i++)
    rows[i] = copy_from[i];
  return *this;
}

template <int M, int N, typename T, typename T2>
inline Point<M, T> operator*(const Matrix<M, N, T> &m, const Point<N, T2> &p)
{
  Point<M, T> out;
  for(int j = 0; j < M; j++)
    out[j] = m.rows[j].dot(p);
  return out;
}

template <int M, int P, int N, typename T, typename T2>
inline Matrix<M, N, T> operator*(const Matrix<M, P, T> &m, const Matrix<P, N, T2> &n)
{
  Matrix<M, N, T> out;
  for(int i = 0; i < M; i++)
    for(int j = 0; j < N; j++) {
      out[i][j] = m[i][0] * n[0][j];
      for(int k = 1; k < P; k++)
        out[i][j] += m[i][k] * n[k][j];
    }
  return out;
}

template <int M, int N, typename T>
inline Point<N, T> &Matrix<M, N, T>::operator[](int index)
{
  return rows[index];
}

template <int M, int N, typename T>
inline const Point<N, T> &Matrix<M, N, T>::operator[](int index) const
{
  return rows[index];
}

////////////////////////////////////////////////////////////////////////
//
// class PointInRectIterator<N,T>

template <int N, typename T>
inline PointInRectIterator<N, T>::PointInRectIterator(void)
  : valid(false)
{}

template <int N, typename T>
inline PointInRectIterator<N, T>::PointInRectIterator(const Rect<N, T> &_r,
                                                      bool _fortran_order /*= true*/)
  : p(_r.lo)
  , valid(!_r.empty())
  , rect(_r)
  , fortran_order(_fortran_order)
{}

template <int N, typename T>
inline void PointInRectIterator<N, T>::reset(const Rect<N, T> &_r,
                                             bool _fortran_order /*= true*/)
{
  p = _r.lo;
  valid = !_r.empty();
  rect = _r;
  fortran_order = _fortran_order;
}

template <int N, typename T>
inline bool PointInRectIterator<N, T>::step(void)
{
  assert(valid); // can't step an iterator that's already done
  if(!valid)
    return false;
    // despite the check above, g++ 11.1 in c++20 mode complains that `rect`
    //  might be uninitialized even though the only way `valid` can become
    //  true is with an initialization of `rect`
#ifdef REALM_COMPILER_IS_GCC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
  if(N == 1) {
    // 1-D doesn't care about fortran/C order
    if(p[0] < rect.hi[0]) {
      p[0]++;
      return true;
    }
  } else {
    if(fortran_order) {
      // do dimensions in increasing order
      for(int i = 0; i < N; i++) {
        if(p[i] < rect.hi[i]) {
          p[i]++;
          return true;
        }
        p[i] = rect.lo[i];
      }
    } else {
      // do dimensions in decreasing order
      for(int i = N - 1; i >= 0; i--) {
        if(p[i] < rect.hi[i]) {
          p[i]++;
          return true;
        }
        p[i] = rect.lo[i];
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////
//
// class AffineAccessor<FT,N,T>

template <typename FT, int N, typename T>
inline AffineAccessor<FT, N, T>::AffineAccessor(void)
{}

template <typename FT, int N, typename T>
inline AffineAccessor<FT, N, T>::~AffineAccessor(void)
{}

template <typename FT, int N, typename T>
inline void AffineAccessor<FT, N, T>::reset()
{
  base = 0;
}
template <typename FT, int N, typename T>
inline FT *AffineAccessor<FT, N, T>::ptr(const Point<N, T> &p) const
{
  return this->get_ptr(p);
}

template <typename FT, int N, typename T>
inline FT AffineAccessor<FT, N, T>::read(const Point<N, T> &p) const
{
  return *(this->get_ptr(p));
}

template <typename FT, int N, typename T>
inline void AffineAccessor<FT, N, T>::write(const Point<N, T> &p, FT newval) const
{
  *(this->get_ptr(p)) = newval;
}

template <typename FT, int N, typename T>
inline FT &AffineAccessor<FT, N, T>::operator[](const Point<N, T> &p) const
{
  return *(this->get_ptr(p));
}

template <typename FT, int N, typename T>
inline bool AffineAccessor<FT, N, T>::is_dense_arbitrary(const Rect<N, T> &bounds) const
{
  size_t exp_offset = sizeof(FT);
  int used_mask = 0; // keep track of which dimensions we've already matched
  for(int i = 0; i < N; i++) {
    bool found = false;
    for(int j = 0; j < N; j++) {
      if((used_mask >> j) & 1)
        continue;
      if(strides[j] != exp_offset) {
        // Mask off any dimensions with stride 0
        if(strides[j] == 0) {
          if(bounds.lo[j] != bounds.hi[j])
            return false;
          used_mask |= (1 << j);
          if(++i == N) {
            found = true;
            break;
          }
        }
        continue;
      }
      found = true;
      // It's possible other dimensions can have the same strides if
      // there are multiple dimensions with extents of size 1. At most
      // one dimension must have an extent >1 though
      int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
      for(int k = j + 1; k < N; k++) {
        if((used_mask >> k) & 1)
          continue;
        if(strides[k] == exp_offset) {
          if(bounds.lo[k] < bounds.hi[k]) {
            if(nontrivial >= 0) // if we already saw a non-trivial dimension this is bad
              return false;
            else
              nontrivial = k;
          }
          used_mask |= (1 << k);
          i++;
        }
      }
      used_mask |= (1 << j);
      if(nontrivial >= 0)
        exp_offset *= (bounds.hi[nontrivial] - bounds.lo[nontrivial] + 1);
      break;
    }
    if(!found)
      return false;
  }
  return true;
}

template <typename FT, int N, typename T>
inline bool AffineAccessor<FT, N, T>::is_dense_col_major(const Rect<N, T> &bounds) const
{
  size_t exp_offset = sizeof(FT);
  for(int i = 0; i < N; i++) {
    if(strides[i] != exp_offset) {
      // Special case for stride of zero for unit dimension
      if((strides[i] == 0) && (bounds.lo[i] == bounds.hi[i]))
        continue;
      return false;
    }
    exp_offset *= (bounds.hi[i] - bounds.lo[i] + 1);
  }
  return true;
}

template <typename FT, int N, typename T>
inline bool AffineAccessor<FT, N, T>::is_dense_row_major(const Rect<N, T> &bounds) const
{
  size_t exp_offset = sizeof(FT);
  for(int i = N - 1; i >= 0; i--) {
    if(strides[i] != exp_offset) {
      // Special case for stride of zero for unit dimension
      if((strides[i] == 0) && (bounds.lo[i] == bounds.hi[i]))
        continue;
      return false;
    }
    exp_offset *= (bounds.hi[i] - bounds.lo[i] + 1);
  }
  return true;
}

template <typename FT, int N, typename T>
inline FT *AffineAccessor<FT, N, T>::get_ptr(const Point<N, T> &p) const
{
  uintptr_t rawptr = base;
  for(int i = 0; i < N; i++)
    rawptr += p[i] * strides[i];
  return reinterpret_cast<FT *>(rawptr);
}
