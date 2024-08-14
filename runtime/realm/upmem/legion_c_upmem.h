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

#ifndef LEGION_UPMEM_H
#define LEGION_UPMEM_H

#include "upmem_common.h"

#include "realm/upmem/realm_c_upmem.h"

#define LEGION_MAX_DIM 3

typedef long long coord_t;
typedef size_t FieldID;
typedef size_t DomainPoint;

typedef enum legion_privilege_mode_t
{
  LEGION_NO_ACCESS = 0x00000000,
  LEGION_READ_PRIV = 0x00000001,
  LEGION_READ_ONLY = 0x00000001, // READ_PRIV,
  LEGION_WRITE_PRIV = 0x00000002,
  LEGION_REDUCE_PRIV = 0x00000004,
  LEGION_REDUCE = 0x00000004,              // REDUCE_PRIV,
  LEGION_READ_WRITE = 0x00000007,          // READ_PRIV | WRITE_PRIV | REDUCE_PRIV,
  LEGION_DISCARD_MASK = 0x10000000,        // For marking we don't need inputs
  LEGION_DISCARD_INPUT_MASK = 0x10000000,  // Rename of the old discard mask
  LEGION_DISCARD_OUTPUT_MASK = 0x20000000, // For marking we don't need outputs
  LEGION_WRITE_ONLY = 0x10000002,          // WRITE_PRIV | DISCARD_INPUT_MASK
  LEGION_WRITE_DISCARD = 0x10000007,       // READ_WRITE | DISCARD_INPUT_MASK
} PrivilegeMode;

namespace Legion {

  /**
   * \class FieldAccessor
   * A field accessor is a class used to get access to the data
   * inside of a PhysicalRegion object for a specific field. The
   * default version of this class is empty, but the following
   * specializations of this class with different privilege modes
   * will provide different methods specific to that privilege type
   * The ReduceAccessor class should be used for explicit reductions
   *
   * READ_ONLY
   *  - FT read(const Point<N,T>&) const
   *  ------ Methods below here for [Multi-]Affine Accessors only ------
   *  - const FT* ptr(const Point<N,T>&) const
   *  - const FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (dense)
   *  - const FT* ptr(const Rect<N,T>&, size_t strides[N],
   *                  size_t=sizeof(FT)) const
   *  - const FT& operator[](const Point<N,T>&) const
   *
   * READ_WRITE
   *  - FT read(const Point<N,T>&) const
   *  - void write(const Point<N,T>&, FT val) const
   *  ------ Methods below here for [Multi-]Affine Accessors only ------
   *  - FT* ptr(const Point<N,T>&) const
   *  - FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (must be dense)
   *  - FT* ptr(const Rect<N,T>&, size_t strides[N], size_t=sizeof(FT)) const
   *  - FT& operator[](const Point<N,T>&) const
   *  - template<typename REDOP, bool EXCLUSIVE>
   *      void reduce(const Point<N,T>&, REDOP::RHS) const
   *
   *  WRITE_DISCARD
   *  - void write(const Point<N,T>&, FT val) const
   *  ------ Methods below here for [Multi-]Affine Accessors only ------
   *  - FT* ptr(const Point<N,T>&) const
   *  - FT* ptr(const Rect<N,T>&, size_t = sizeof(FT)) const (must be dense)
   *  - FT* ptr(const Rect<N,T>&, size_t strides[N], size_t=sizeof(FT)) const
   *  - FT& operator[](const Point<N,T>&) const
   */
  template <PrivilegeMode MODE, typename FT, int N, typename COORD_T = coord_t,
            typename A = Realm::AffineAccessor<FT, N, COORD_T>,
#ifdef LEGION_BOUNDS_CHECKS
            bool CHECK_BOUNDS = true>
#else
            bool CHECK_BOUNDS = false>
#endif
  class FieldAccessor {
  private:
    static_assert(N > 0, "N must be positive");

  public:
    FieldAccessor(void) {}

  public:
    typedef FT value_type;
    typedef FT &reference;
    typedef const FT &const_reference;
    static const int dim = N;
  };

  template <int DIM, typename COORD_T = coord_t>
  class PointInRectIterator {
  private:
    static_assert(DIM > 0, "DIM must be positive");

  public:
    PointInRectIterator(void);
    PointInRectIterator(const Rect<DIM, COORD_T> &r, bool column_major_order = true);

  public:
    inline bool valid(void) const;
    inline bool step(void);

  public:
    inline bool operator()(void) const;
    inline Point<DIM, COORD_T> operator*(void) const;
    inline COORD_T operator[](unsigned index) const;
    inline const Point<DIM, COORD_T> *operator->(void) const;
    inline PointInRectIterator<DIM, COORD_T> &operator++(void);
    inline PointInRectIterator<DIM, COORD_T> operator++(int /*postfix*/);

  protected:
    Realm::PointInRectIterator<DIM, COORD_T> itr;
  };

}; // namespace Legion

#include "realm/upmem/legion_c_upmem.inl"

#endif