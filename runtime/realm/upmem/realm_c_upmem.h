/* Copyright 2024 Stanford University, NVIDIA Corporation
 *                Los Alamos National Laboratory, Northwestern University
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef REALM_UPMEM_H
#define REALM_UPMEM_H

#include "upmem_common.h"

extern "C" {
#include <stdint.h>
#include <defs.h>
#include <mram.h>
#include <alloc.h>
#include <barrier.h>
#include <assert.h>
}

#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif

EXTERNC inline int __cxa_atexit(void (*func)(void *), void *arg, void *d) { return 0; }

namespace Realm {

  template <int N, typename T = int>
  struct Point;
  template <int N, typename T = int>
  struct Rect;
  template <int N, typename T = int>
  class PointInRectIterator;
  template <int M, int N, typename T = int>
  struct Matrix;

  // a Point is a tuple describing a point in an N-dimensional space - the default "base
  // type"
  //  for each dimension is int, but 64-bit indices are supported as well
  template <int N, typename T>
  struct __attribute__((aligned(8))) Point {

    typedef T value_type;
    value_type values[N]; // use operator[] instead

    Point(void) = default;

    explicit Point(value_type val);

    template <typename Arg0, typename Arg1, typename... Args>
    Point(Arg0 val0, Arg1 val1, Args... vals);
    // construct from any integral value
    template <typename T2>
    explicit Point(T2 val, T2); // same value for all dimensions
    template <typename T2>
    explicit Point(T2 vals[N], T2);

    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>
    Point(const Point<N, T2> &copy_from);
    template <typename T2>
    Point<N, T> &operator=(const Point<N, T2> &copy_from);

    T &operator[](int index);

    const T &operator[](int index) const;

    template <typename T2>
    T dot(const Point<N, T2> &rhs) const;

    // 1-4D accessors.  These will only be available if the class's dimensioned allow for
    // it, otherwise it is a compiler error to use them
    T &x();
    T &y();
    T &z();
    T &w();
    const T &x() const;
    const T &y() const;
    const T &z() const;
    const T &w() const;

    static constexpr Point<N, T> ZEROES(void);

    static constexpr Point<N, T> ONES(void);
  };

  // component-wise operators defined on Point<N,T> (with optional coercion)
  template <int N, typename T, typename T2>
  bool operator==(const Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  bool operator!=(const Point<N, T> &lhs, const Point<N, T2> &rhs);

  template <int N, typename T, typename T2>
  Point<N, T> operator+(const Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> &operator+=(Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> operator-(const Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> &operator-=(Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> operator*(const Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> &operator*=(Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> operator/(const Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> &operator/=(Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> operator%(const Point<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Point<N, T> &operator%=(Point<N, T> &lhs, const Point<N, T2> &rhs);

  // a Rect is a pair of points defining the lower and upper bounds of an N-D rectangle
  //  the bounds are INCLUSIVE

  template <int N, typename T>
  struct __attribute__((aligned(8))) Rect {
    Point<N, T> lo, hi;

    Rect(void);

    Rect(const Point<N, T> &_lo, const Point<N, T> &_hi);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>

    Rect(const Rect<N, T2> &copy_from);
    template <typename T2>

    Rect<N, T> &operator=(const Rect<N, T2> &copy_from);

    // constructs a guaranteed-empty rectangle

    static Rect<N, T> make_empty(void);

    bool empty(void) const;

    size_t volume(void) const;

    bool contains(const Point<N, T> &p) const;

    // true if all points in other are in this rectangle

    bool contains(const Rect<N, T> &other) const;

    // bool contains(const IndexSpace<N, T> &is) const;

    // true if there are any points in the intersection of the two rectangles

    bool overlaps(const Rect<N, T> &other) const;

    Rect<N, T> intersection(const Rect<N, T> &other) const;

    // returns the _bounding box_ of the union of two rectangles (the actual union
    //  might not be a rectangle)

    Rect<N, T> union_bbox(const Rect<N, T> &other) const;

    template <int N2, typename T2>
    Rect<N2, T2> apply_transform(const Matrix<N2, N, T2> &transform,
                                 const Point<N2, T2> &offset) const;

    /*
    // copy and fill operations (wrappers for IndexSpace versions)
    Event fill(const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               const void *fill_value, size_t fill_value_size,
               Event wait_on = Event::NO_EVENT,
               int priority = 0) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               int priority = 0) const;

    Event copy(const std::vector<CopySrcDstField> &srcs,
               const std::vector<CopySrcDstField> &dsts,
               const IndexSpace<N,T> &mask,
               const ProfilingRequestSet &requests,
               Event wait_on = Event::NO_EVENT,
               int priority = 0) const;

    */
  };

  template <int N, typename T, typename T2>
  bool operator==(const Rect<N, T> &lhs, const Rect<N, T2> &rhs);
  template <int N, typename T, typename T2>
  bool operator!=(const Rect<N, T> &lhs, const Rect<N, T2> &rhs);

  // rectangles may be displaced by a vector (i.e. point)
  template <int N, typename T, typename T2>
  Rect<N, T> operator+(const Rect<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Rect<N, T> &operator+=(Rect<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Rect<N, T> operator-(const Rect<N, T> &lhs, const Point<N, T2> &rhs);
  template <int N, typename T, typename T2>
  Rect<N, T> &operator-=(Rect<N, T> &lhs, const Rect<N, T2> &rhs);

  template <int M, int N, typename T>
  struct __attribute__((aligned(8))) Matrix {
    Point<N, T> rows[M];

    Matrix(void);
    // copies allow type coercion (assuming the underlying type does)
    template <typename T2>

    Matrix(const Matrix<M, N, T2> &copy_from);
    template <typename T2>

    Matrix<M, N, T> &operator=(const Matrix<M, N, T2> &copy_from);

    Point<N, T> &operator[](int index);

    const Point<N, T> &operator[](int index) const;
  };

  template <int M, int N, typename T, typename T2>
  Point<M, T> operator*(const Matrix<M, N, T> &m, const Point<N, T2> &p);
  template <int M, int P, int N, typename T, typename T2>
  Matrix<M, N, T> operator*(const Matrix<M, P, T> &m, const Matrix<P, N, T2> &n);

  template <int N, typename T>
  class __attribute__((aligned(8))) PointInRectIterator {
  public:
    Point<N, T> p;
    bool valid;
    Rect<N, T> rect;
    bool fortran_order;

    PointInRectIterator(void);

    PointInRectIterator(const Rect<N, T> &_r, bool _fortran_order = true);

    void reset(const Rect<N, T> &_r, bool _fortran_order = true);

    bool step(void);
  };

  /**
   * \class AffineAccessor
   * An accessor that works for any instance layout that can be
   * linearized using an affine transformation.  AffineAccessors are
   * generally more efficient than GenericAccessors, but are only
   * applicable to a subset of instance layouts.
   * The random-access look-ups are O(1) in the size of the instance.
   */
  template <typename FT, int N, typename T = int>
  class __attribute__((aligned(8))) AffineAccessor {
  public:
    // NOTE: even when compiling with nvcc, non-default constructors are only
    //  available in host code

    // TODO: Sean check if this is safe for a default constructor
    AffineAccessor(void);

    // /**
    //  * Construct an AffineAccessor for the specified instance and field.
    //  * Implicitly tries to cover the entire instance's domain.
    //  * NOTE: this constructor will die if the instance is not compatible
    //  * with AffineAccessor. Call is_compatible(...) first if you're not
    //  * sure.
    //  * \param inst RegionInstance to construct accessor for.
    //  * \param field_id FieldID of field to construct accessor for.
    //  * \param subfield_offset Offset of subfield to construct accessor for.
    //  */
    // AffineAccessor(RegionInstance inst,
    //    FieldID field_id, size_t subfield_offset = 0);

    // /** Construct an AffineAccessor for the specified instance and field.
    //  * Limits the domain to the specified subrectangle.
    //  * NOTE: this constructor will die if the instance is not compatible
    //  * with AffineAccessor. Call is_compatible(...) first if you're not
    //  * sure.
    //  * \param inst RegionInstance to construct accessor for.
    //  * \param field_id FieldID of field to construct accessor for.
    //  * \param subrect Subrectangle to limit domain to.
    //  * \param subfield_offset Offset of subfield to construct accessor for.
    //  */
    // AffineAccessor(RegionInstance inst,
    //    FieldID field_id, const Rect<N,T>& subrect,
    //    size_t subfield_offset = 0);

    // ///@{
    // /**
    //  * Construct an AffineAccessor for the specified instance, field,
    //  * and coordinate transform.
    //  * \param inst RegionInstance to construct accessor for.
    //  * \param transform Affine transform to apply to coordinates.
    //  * \param offset Offset to apply to coordinates.
    //  * \param field_id FieldID of field to construct accessor for.
    //  * \param subfield_offset Offset of subfield to construct accessor for.
    //  */
    // template <int N2, typename T2>
    // AffineAccessor(RegionInstance inst,
    //    const Matrix<N2, N, T2>& transform,
    //    const Point<N2, T2>& offset,
    //    FieldID field_id, size_t subfield_offset = 0);

    // // note that the subrect here is in in the accessor's indexspace
    // //  (from which the corresponding subrectangle in the instance can be
    // //  easily determined)
    // template <int N2, typename T2>
    // AffineAccessor(RegionInstance inst,
    //    const Matrix<N2, N, T2>& transform,
    //    const Point<N2, T2>& offset,
    //    FieldID field_id, const Rect<N,T>& subrect,
    //    size_t subfield_offset = 0);
    // ///@}

    ~AffineAccessor(void);

    AffineAccessor(const AffineAccessor &) = default;
    AffineAccessor &operator=(const AffineAccessor &) = default;
    AffineAccessor(AffineAccessor &&) noexcept = default;
    AffineAccessor &operator=(AffineAccessor &&) noexcept = default;
    /*
    static bool is_compatible(RegionInstance inst, FieldID field_id);
    static bool is_compatible(RegionInstance inst, FieldID field_id, const Rect<N,T>&
    subrect); template <int N2, typename T2> static bool is_compatible(RegionInstance
    inst, const Matrix<N2, N, T2>& transform, const Point<N2, T2>& offset, FieldID
    field_id); template <int N2, typename T2> static bool is_compatible(RegionInstance
    inst, const Matrix<N2, N, T2>& transform, const Point<N2, T2>& offset, FieldID
    field_id, const Rect<N,T>& subrect);
    */

    // used by constructors or can be called directly
    void reset();
    /*
    void reset(RegionInstance inst,
               FieldID field_id, size_t subfield_offset = 0);
    void reset(RegionInstance inst,
               FieldID field_id, const Rect<N,T>& subrect,
               size_t subfield_offset = 0);
    template <int N2, typename T2>
    void reset(RegionInstance inst,
               const Matrix<N2, N, T2>& transform,
               const Point<N2, T2>& offset,
               FieldID field_id, size_t subfield_offset = 0);
    template <int N2, typename T2>
    void reset(RegionInstance inst,
               const Matrix<N2, N, T2>& transform,
               const Point<N2, T2>& offset,
               FieldID field_id, const Rect<N,T>& subrect,
               size_t subfield_offset = 0);
    */

    FT *ptr(const Point<N, T> &p) const;
    FT read(const Point<N, T> &p) const;
    void write(const Point<N, T> &p, FT newval) const;

    FT &operator[](const Point<N, T> &p) const;

    bool is_dense_arbitrary(const Rect<N, T> &bounds) const; // any dimension ordering
    bool is_dense_col_major(const Rect<N, T> &bounds) const; // Fortran dimension ordering
    bool is_dense_row_major(const Rect<N, T> &bounds) const; // C dimension ordering

    uintptr_t base;
    Point<N, size_t> strides;

  protected:
    FT *get_ptr(const Point<N, T> &p) const;
  };

}; // namespace Realm
#include "realm/upmem/realm_c_upmem.inl"

#endif