using namespace Legion;

//----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline Legion::PointInRectIterator<DIM,COORD_T>::PointInRectIterator(void)
  //----------------------------------------------------------------------------
  {
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline Legion::PointInRectIterator<DIM,COORD_T>::PointInRectIterator(
             const Rect<DIM,COORD_T> &r, bool column_major_order)
    : itr(Realm::PointInRectIterator<DIM,COORD_T>(r, column_major_order))
  //----------------------------------------------------------------------------
  {
    assert(valid());
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline bool Legion::PointInRectIterator<DIM,COORD_T>::valid(void) const
  //----------------------------------------------------------------------------
  {
    return itr.valid;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline bool Legion::PointInRectIterator<DIM,COORD_T>::step(void)
  //----------------------------------------------------------------------------
  {
    assert(valid());
    itr.step();
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline bool Legion::PointInRectIterator<DIM,COORD_T>::operator()(void) const
  //----------------------------------------------------------------------------
  {
    return valid();
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline Point<DIM,COORD_T> 
                         Legion::PointInRectIterator<DIM,COORD_T>::operator*(void) const
  //----------------------------------------------------------------------------
  {
    return itr.p;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline COORD_T 
              Legion::PointInRectIterator<DIM,COORD_T>::operator[](unsigned index) const
  //----------------------------------------------------------------------------
  {
    return itr.p[index];
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline const Point<DIM,COORD_T>* 
                        Legion::PointInRectIterator<DIM,COORD_T>::operator->(void) const
  //----------------------------------------------------------------------------
  {
    return &(itr.p);
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline Legion::PointInRectIterator<DIM,COORD_T>&
                              Legion::PointInRectIterator<DIM,COORD_T>::operator++(void)
  //----------------------------------------------------------------------------
  {
    step();
    return *this;
  }

  //----------------------------------------------------------------------------
  template<int DIM, typename COORD_T> 
  inline Legion::PointInRectIterator<DIM,COORD_T>
                    Legion::PointInRectIterator<DIM,COORD_T>::operator++(int/*postfix*/)
  //----------------------------------------------------------------------------
  {
    Legion::PointInRectIterator<DIM,COORD_T> result(*this);
    step();
    return result;
  }


// Some helper methods for accessors and deferred buffers
namespace Internal {
  template <int N, typename T>
  static inline bool is_dense_layout(const Rect<N, T> &bounds, const size_t strides[N],
                                     size_t field_size)
  {
    ptrdiff_t exp_offset = field_size;
    int used_mask = 0; // keep track of the dimensions we've already matched
    static_assert((N <= (8 * sizeof(used_mask))), "Mask dim exceeded");
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
        // one dimension can have an extent >1 though
        int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
        for(int k = j + 1; k < N; k++) {
          if((used_mask >> k) & 1)
            continue;
          if(strides[k] == exp_offset) {
            if(bounds.lo[k] < bounds.hi[k]) {
              // if we already saw a non-trivial dimension this is bad
              if(nontrivial >= 0)
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

  // Same method as above but for realm points from affine accessors
  template <int N, typename T>
  static inline bool is_dense_layout(const Rect<N, T> &bounds,
                                     const Realm::Point<N, size_t> &strides,
                                     size_t field_size)
  {
    size_t exp_offset = field_size;
    int used_mask = 0; // keep track of the dimensions we've already matched
    static_assert((N <= (8 * sizeof(used_mask))), "Mask dim exceeded");
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
        // one dimension can have an extent >1 though
        int nontrivial = (bounds.lo[j] < bounds.hi[j]) ? j : -1;
        for(int k = j + 1; k < N; k++) {
          if((used_mask >> k) & 1)
            continue;
          if(strides[k] == exp_offset) {
            if(bounds.lo[k] < bounds.hi[k]) {
              // if we already saw a non-trivial dimension this is bad
              if(nontrivial >= 0)
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
}; // namespace Internal
   // Special namespace for providing multi-dimensional
   // array syntax on accessors
namespace ArraySyntax {
  // A helper class for handling reductions
  template <typename A, typename FT, int N, typename T>
  class ReductionHelper {
  public:
    ReductionHelper(const A &acc, const Point<N> &p)
      : accessor(acc)
      , point(p)
    {}

  public:
    inline void reduce(FT val) const { accessor.reduce(point, val); }
    inline void operator<<=(FT val) const { accessor.reduce(point, val); }

  public:
    const A &accessor;
    const Point<N, T> point;
  };

  template <typename FT, PrivilegeMode P>
  class AccessorRefHelper {
  public:
    AccessorRefHelper(const Realm::AccessorRefHelper<FT> &h)
      : helper(h)
    {}

  public:
    // read
    inline operator FT(void) const { return helper; }
    // writes
    inline AccessorRefHelper<FT, P> &operator=(const FT &newval)
    {
      helper = newval;
      return *this;
    }
    template <PrivilegeMode P2>
    inline AccessorRefHelper<FT, P> &operator=(const AccessorRefHelper<FT, P2> &rhs)
    {
      helper = rhs.helper;
      return *this;
    }

  protected:
    template <typename T, PrivilegeMode P2>
    friend class AccessorRefHelper;
    Realm::AccessorRefHelper<FT> helper;
  };

  template <typename FT>
  class AccessorRefHelper<FT, LEGION_READ_ONLY> {
  public:
    AccessorRefHelper(const Realm::AccessorRefHelper<FT> &h)
      : helper(h)
    {}
    // read
    inline operator FT(void) const { return helper; }

  private:
    // no writes allowed
    inline AccessorRefHelper<FT, LEGION_READ_ONLY> &
    operator=(const AccessorRefHelper<FT, LEGION_READ_ONLY> &rhs)
    {
      helper = rhs.helper;
      return *this;
    }

  protected:
    template <typename T, PrivilegeMode P2>
    friend class AccessorRefHelper;
    Realm::AccessorRefHelper<FT> helper;
  };

  // LEGION_NO_ACCESS means we dynamically check the privilege
  template <typename FT>
  class AccessorRefHelper<FT, LEGION_NO_ACCESS> {
  public:
    AccessorRefHelper(const Realm::AccessorRefHelper<FT> &h, FieldID fid,
                      const DomainPoint &pt, PrivilegeMode p)
      : helper(h)
      , point(pt)
      , field(fid)
      , privilege(p)
    {}

  public:
    // read
    inline operator FT(void) const
    {
      if((privilege & LEGION_READ_PRIV) == 0) {
        assert(0 && "Failed Privilege Check. Not Implemeneted");
        // PhysicalRegion::fail_privilege_check(point, field, privilege);
      }
      return helper;
    }
    // writes
    inline AccessorRefHelper<FT, LEGION_NO_ACCESS> &operator=(const FT &newval)
    {
      if((privilege & LEGION_WRITE_PRIV) == 0) {
        assert(0 && "Failed Privilege Check. Not Implemeneted");
        // PhysicalRegion::fail_privilege_check(point, field, privilege);
      }
      helper = newval;
      return *this;
    }
    template <PrivilegeMode P2>
    inline AccessorRefHelper<FT, LEGION_NO_ACCESS> &
    operator=(const AccessorRefHelper<FT, P2> &rhs)
    {
      if((privilege & LEGION_WRITE_PRIV) == 0) {
        assert(0 && "Failed Privilege Check. Not Implemeneted");
        // PhysicalRegion::fail_privilege_check(point, field, privilege);
      }
      helper = rhs.helper;
      return *this;
    }

  protected:
    template <typename T, PrivilegeMode P2>
    friend class AccessorRefHelper;
    Realm::AccessorRefHelper<FT> helper;
    DomainPoint point;
    FieldID field;
    PrivilegeMode privilege;
  };

  // A small helper class that helps provide some syntactic sugar for
  // indexing accessors like a multi-dimensional array for generic accessors
  template <typename A, typename FT, int N, typename T, int M, PrivilegeMode P>
  class GenericSyntaxHelper {
  public:
    GenericSyntaxHelper(const A &acc, const Point<M - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (M - 1); i++)
        point[i] = p[i];
    }

  public:
    inline GenericSyntaxHelper<A, FT, N, T, M + 1, P> operator[](T val)
    {
      point[M - 1] = val;
      return GenericSyntaxHelper<A, FT, N, T, M + 1, P>(accessor, point);
    }

  public:
    const A &accessor;
    Point<M, T> point;
  };
  // Specialization for M = N
  template <typename A, typename FT, int N, typename T, PrivilegeMode P>
  class GenericSyntaxHelper<A, FT, N, T, N, P> {
  public:
    GenericSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline AccessorRefHelper<FT, P> operator[](T val)
    {
      point[N - 1] = val;
      return accessor[point];
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };
  // Further specialization for M = N and read-only
  template <typename A, typename FT, int N, typename T>
  class GenericSyntaxHelper<A, FT, N, T, N, LEGION_READ_ONLY> {
  public:
    GenericSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline AccessorRefHelper<FT, LEGION_READ_ONLY> operator[](T val)
    {
      point[N - 1] = val;
      return accessor[point];
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };
  // Further specialization for M = N and reductions
  template <typename A, typename FT, int N, typename T>
  class GenericSyntaxHelper<A, FT, N, T, N, LEGION_REDUCE> {
  public:
    GenericSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline const ReductionHelper<A, FT, N, T> operator[](T val)
    {
      point[N - 1] = val;
      return ReductionHelper<A, FT, N, T>(accessor, point);
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };

  // A small helper class that helps provide some syntactic sugar for
  // indexing accessors like a multi-dimensional array for affine accessors
  template <typename A, typename FT, int N, typename T, int M, PrivilegeMode P>
  class AffineSyntaxHelper {
  public:
    AffineSyntaxHelper(const A &acc, const Point<M - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (M - 1); i++)
        point[i] = p[i];
    }

  public:
    inline AffineSyntaxHelper<A, FT, N, T, M + 1, P> operator[](T val)
    {
      point[M - 1] = val;
      return AffineSyntaxHelper<A, FT, N, T, M + 1, P>(accessor, point);
    }

  public:
    const A &accessor;
    Point<M, T> point;
  };

  // Specialization for M = N
  template <typename A, typename FT, int N, typename T, PrivilegeMode P>
  class AffineSyntaxHelper<A, FT, N, T, N, P> {
  public:
    AffineSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline FT &operator[](T val)
    {
      point[N - 1] = val;
      return accessor[point];
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };

  // Further specialization for M = N and read-only
  template <typename A, typename FT, int N, typename T>
  class AffineSyntaxHelper<A, FT, N, T, N, LEGION_READ_ONLY> {
  public:
    AffineSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline const FT &operator[](T val)
    {
      point[N - 1] = val;
      return accessor[point];
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };

  // Further specialize for M = N and reductions
  template <typename A, typename FT, int N, typename T>
  class AffineSyntaxHelper<A, FT, N, T, N, LEGION_REDUCE> {
  public:
    AffineSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline const ReductionHelper<A, FT, N, T> operator[](T val)
    {
      point[N - 1] = val;
      return ReductionHelper<A, FT, N, T>(accessor, point);
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };

  // Helper class for affine syntax that behaves like a
  // pointer/reference, but does dynamic privilege checks
  template <typename FT>
  class AffineRefHelper {
  public:
    AffineRefHelper(FT &r, FieldID fid, const DomainPoint &pt, PrivilegeMode p)
      : ref(r)
      , privilege(p)
    {}

  public:
    // read
    inline operator const FT &(void) const
    {
      assert(privilege & LEGION_READ_PRIV);
      return ref;
    }
    // writes
    inline AffineRefHelper<FT> &operator=(const FT &newval)
    {
      assert(privilege & LEGION_WRITE_PRIV);
      ref = newval;
      return *this;
    }
    inline AffineRefHelper<FT> &operator=(const AffineRefHelper<FT> &rhs)
    {
      assert(privilege & LEGION_WRITE_PRIV);
      ref = rhs.ref;
      return *this;
    }

  protected:
    FT &ref;
    PrivilegeMode privilege;
  };

  // Further specialization for M = N and NO_ACCESS (dynamic privilege)
  template <typename A, typename FT, int N, typename T>
  class AffineSyntaxHelper<A, FT, N, T, N, LEGION_NO_ACCESS> {
  public:
    AffineSyntaxHelper(const A &acc, const Point<N - 1, T> &p)
      : accessor(acc)
    {
      for(int i = 0; i < (N - 1); i++)
        point[i] = p[i];
    }

  public:
    inline AffineRefHelper<FT> operator[](T val)
    {
      point[N - 1] = val;
      return accessor[point];
    }

  public:
    const A &accessor;
    Point<N, T> point;
  };
}; // namespace ArraySyntax
   
 // Read-only FieldAccessor specialization
    template<typename FT, int N, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,CB> {
    private:
      static_assert(N > 0, "DIM must be positive");
    public:
      FieldAccessor(void) { }
    public:
      inline FT read(const Point<N,T>& p) const 
        { 
          return accessor.read(p); 
        }
      
      inline FT operator[](const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }

      // inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY> 
      //     operator[](const Point<N,T>& p) const
      //   { 
      //     return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
      //                                                     accessor[p]);
      //   }
      inline ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
            Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
               Realm::AffineAccessor<FT,N,T>,CB>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::AffineAccessor<FT,N,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization
    // with bounds checks
    template<typename FT, int N, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,N,T,
                        Realm::AffineAccessor<FT,N,T>,true> {
    private:
      static_assert(N > 0, "DIM must be positive");
    public:
      FieldAccessor(void) { }
    public:

          inline FT read(const Point<N,T>& p) const 
        { 
          return accessor.read(p); 
        }
            
      inline FT operator[](const Point<N,T>& p) const
        { 
          return accessor.read(p); 
        }

      // inline FT read(const Point<N,T>& p) const 
      //   { 
      //     if (!bounds.contains(p)) {
      //       assert(0 && "Failed fail_bounds_check. Not Implemeneted");
      //       // PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
      //       //                                   LEGION_READ_ONLY);
      //     }
      //     return accessor.read(p); 
      //   }
      // inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>
      //     operator[](const Point<N,T>& p) const
      //   { 
      //     if (!bounds.contains(p)) {
      //       assert(0 && "Failed fail_bounds_check. Not Implemeneted");
      //       // PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
      //       //                                   LEGION_READ_ONLY);
      //     }
      //     return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
      //                                                     accessor[p]);
      //   }
      inline ArraySyntax::GenericSyntaxHelper<
          FieldAccessor<LEGION_READ_ONLY,FT,N,T,
             Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>
          operator[](T index) const
      {
        return ArraySyntax::GenericSyntaxHelper<
            FieldAccessor<LEGION_READ_ONLY,FT,N,T,
              Realm::AffineAccessor<FT,N,T>,true>,FT,N,T,2,LEGION_READ_ONLY>(
              *this, Point<1,T>(index));
      }
    public:
      mutable Realm::AffineAccessor<FT,N,T> accessor;
      FieldID field;
      Rect<N,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = N;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity
    template<typename FT, typename T, bool CB>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,CB> {
    public:
      FieldAccessor(void) { }
    public:
      inline FT read(const Point<1,T>& p) const 
        { 
          return accessor.read(p); 
        }
            
      inline FT operator[](const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }

      // inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>
      //     operator[](const Point<1,T>& p) const
      //   { 
      //     return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(accessor[p]);
      //   }

    public:
      mutable Realm::AffineAccessor<FT,1,T> accessor;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };

    // Read-only FieldAccessor specialization 
    // with N==1 to avoid array ambiguity and bounds checks
    template<typename FT, typename T>
    class FieldAccessor<LEGION_READ_ONLY,FT,1,T,
                        Realm::AffineAccessor<FT,1,T>,true> {
    public:
      // No CUDA support due to PhysicalRegion constructor
      FieldAccessor(void) { }
    public:

          inline FT read(const Point<1,T>& p) const 
        { 
          return accessor.read(p); 
        }
            
      inline FT operator[](const Point<1,T>& p) const
        { 
          return accessor.read(p); 
        }


      // inline FT read(const Point<1,T>& p) const 
      //   { 
      //     if (!bounds.contains(p)) {
      //       assert(0 && "Failed fail_bounds_check. Not Implemeneted");
      //       // PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
      //       //                                   LEGION_READ_ONLY);
      //     }
      //     return accessor.read(p); 
      //   }
      // inline const ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY> 
      //     operator[](const Point<1,T>& p) const
      //   { 
      //     if (!bounds.contains(p)) {
      //       assert(0 && "Failed fail_bounds_check. Not Implemeneted");
      //       // PhysicalRegion::fail_bounds_check(DomainPoint(p), field, 
      //       //                                   LEGION_READ_ONLY);
      //     }
      //     return ArraySyntax::AccessorRefHelper<FT,LEGION_READ_ONLY>(
      //                                                     accessor[p]);
      //   }
    public:
      mutable Realm::AffineAccessor<FT,1,T> accessor;
      FieldID field;
      Rect<1,T> bounds;
    public:
      typedef FT value_type;
      typedef FT& reference;
      typedef const FT& const_reference;
      static const int dim = 1;
    };
   
// Read-write FieldAccessor specialization
template <typename FT, int N, typename T, bool CB>
class FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
private:
  static_assert(N > 0, "DIM must be positive");

public:
  FieldAccessor(void) {}

public:
  inline FT read(const Point<N, T> &p) const { return accessor.read(p); }
  inline void write(const Point<N, T> &p, FT val) const { accessor.write(p, val); }
  inline FT *ptr(const Point<N, T> &p) const { return accessor.ptr(p); }
  inline FT *ptr(const Rect<N, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t strides[N],
                 size_t field_size = sizeof(FT)) const
  {
    for(int i = 0; i < N; i++)
      strides[i] = accessor.strides[i] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<N, T> &p) const { return accessor[p]; }
  inline ArraySyntax::AffineSyntaxHelper<
      FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>, FT,
      N, T, 2, LEGION_READ_WRITE>
  operator[](T index) const
  {
    return ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
        FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
  }
  template <typename REDOP, bool EXCLUSIVE>
  inline void reduce(const Point<N, T> &p, typename REDOP::RHS val) const
  {
    REDOP::template apply<EXCLUSIVE>(accessor[p], val);
  }

public:
  Realm::AffineAccessor<FT, N, T> accessor;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = N;
};

// Read-write FieldAccessor specialization
// with bounds checks
template <typename FT, int N, typename T>
class FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
private:
  static_assert(N > 0, "DIM must be positive");

public:
  // No CUDA support due to PhysicalRegion constructor
  FieldAccessor(void) {}

public:
  inline FT read(const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.read(p);
  }
  inline void write(const Point<N, T> &p, FT val) const
  {
    assert(bounds.contains(p));
    accessor.write(p, val);
  }
  inline FT *ptr(const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.ptr(p);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t strides[N],
                 size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    for(int i = 0; i < N; i++)
      strides[i] = accessor.strides[i] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor[p];
  }
  inline ArraySyntax::AffineSyntaxHelper<
      FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, true>,
      FT, N, T, 2, LEGION_READ_WRITE>
  operator[](T index) const
  {
    return ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_READ_WRITE, FT, N, T, Realm::AffineAccessor<FT, N, T>, true>,
        FT, N, T, 2, LEGION_READ_WRITE>(*this, Point<1, T>(index));
  }
  template <typename REDOP, bool EXCLUSIVE>
  inline void reduce(const Point<N, T> &p, typename REDOP::RHS val) const
  {
    assert(bounds.contains(p));
    REDOP::template apply<EXCLUSIVE>(accessor[p], val);
  }

public:
  Realm::AffineAccessor<FT, N, T> accessor;
  FieldID field;
  Rect<N, T> bounds;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = N;
};

// Read-write FieldAccessor specialization
// with N==1 to avoid array ambiguity
template <typename FT, typename T, bool CB>
class FieldAccessor<LEGION_READ_WRITE, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
public:
  FieldAccessor(void) {}

public:
  inline FT read(const Point<1, T> &p) const { return accessor.read(p); }
  inline void write(const Point<1, T> &p, FT val) const { accessor.write(p, val); }
  inline FT *ptr(const Point<1, T> &p) const { return accessor.ptr(p); }
  inline FT *ptr(const Rect<1, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t strides[1],
                 size_t field_size = sizeof(FT)) const
  {
    strides[0] = accessor.strides[0] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<1, T> &p) const { return accessor[p]; }
  template <typename REDOP, bool EXCLUSIVE>
  inline void reduce(const Point<1, T> &p, typename REDOP::RHS val) const
  {
    REDOP::template apply<EXCLUSIVE>(accessor[p], val);
  }

public:
  Realm::AffineAccessor<FT, 1, T> accessor;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = 1;
};

// Read-write FieldAccessor specialization
// with N==1 to avoid array ambiguity and bounds checks
template <typename FT, typename T>
class FieldAccessor<LEGION_READ_WRITE, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
public:
  // No CUDA support due to PhysicalRegion constructor
  FieldAccessor(void) {}

public:
  inline FT read(const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.read(p);
  }
  inline void write(const Point<1, T> &p, FT val) const
  {
    assert(bounds.contains(p));
    accessor.write(p, val);
  }
  inline FT *ptr(const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.ptr(p);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t strides[1],
                 size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    strides[0] = accessor.strides[0] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor[p];
  }
  template <typename REDOP, bool EXCLUSIVE>
  inline void reduce(const Point<1, T> &p, typename REDOP::RHS val) const
  {
    assert(bounds.contains(p));
    REDOP::template apply<EXCLUSIVE>(accessor[p], val);
  }

public:
  Realm::AffineAccessor<FT, 1, T> accessor;
  FieldID field;
  Rect<1, T> bounds;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = 1;
};

// Write-discard FieldAccessor specialization
template <typename FT, int N, typename T, bool CB>
class FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
private:
  static_assert(N > 0, "DIM must be positive");

public:
  FieldAccessor(void) {}

public:
  inline FT read(const Point<N, T> &p) const { return accessor.read(p); }
  inline void write(const Point<N, T> &p, FT val) const { accessor.write(p, val); }
  inline FT *ptr(const Point<N, T> &p) const { return accessor.ptr(p); }
  inline FT *ptr(const Rect<N, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t strides[N],
                 size_t field_size = sizeof(FT)) const
  {
    for(int i = 0; i < N; i++)
      strides[i] = accessor.strides[i] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<N, T> &p) const { return accessor[p]; }
  inline ArraySyntax::AffineSyntaxHelper<
      FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
      FT, N, T, 2, LEGION_WRITE_DISCARD>
  operator[](T index) const
  {
    return ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                      CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
  }

public:
  Realm::AffineAccessor<FT, N, T> accessor;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = N;
};

// Write-discard FieldAccessor specialization
// with bounds checks
template <typename FT, int N, typename T>
class FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                    true> {
private:
  static_assert(N > 0, "DIM must be positive");

public:
  // No CUDA support due to PhysicalRegion constructor
  FieldAccessor(void) {}

public:
  inline FT read(const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.read(p);
  }
  inline void write(const Point<N, T> &p, FT val) const
  {
    assert(bounds.contains(p));
    accessor.write(p, val);
  }
  inline FT *ptr(const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.ptr(p);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t strides[N],
                 size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    for(int i = 0; i < N; i++)
      strides[i] = accessor.strides[i] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor[p];
  }
  inline ArraySyntax::AffineSyntaxHelper<
      FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                    true>,
      FT, N, T, 2, LEGION_WRITE_DISCARD>
  operator[](T index) const
  {
    return ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                      true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
  }

public:
  Realm::AffineAccessor<FT, N, T> accessor;
  FieldID field;
  Rect<N, T> bounds;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = N;
};

// Write-discard FieldAccessor specialization with
// N == 1 to avoid array ambiguity
template <typename FT, typename T, bool CB>
class FieldAccessor<LEGION_WRITE_DISCARD, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
public:
  FieldAccessor(void) {}

public:
  inline FT read(const Point<1, T> &p) const { return accessor.read(p); }
  inline void write(const Point<1, T> &p, FT val) const { accessor.write(p, val); }
  inline FT *ptr(const Point<1, T> &p) const { return accessor.ptr(p); }
  inline FT *ptr(const Rect<1, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t strides[1],
                 size_t field_size = sizeof(FT)) const
  {
    strides[0] = accessor.strides[0] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<1, T> &p) const { return accessor[p]; }

public:
  Realm::AffineAccessor<FT, 1, T> accessor;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = 1;
};

// Write-discard FieldAccessor specialization with
// N == 1 to avoid array ambiguity and bounds checks
template <typename FT, typename T>
class FieldAccessor<LEGION_WRITE_DISCARD, FT, 1, T, Realm::AffineAccessor<FT, 1, T>,
                    true> {
public:
  // No CUDA support due to PhysicalRegion constructor
  FieldAccessor(void) {}

public:
  inline FT read(const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.read(p);
  }
  inline void write(const Point<1, T> &p, FT val) const
  {
    assert(bounds.contains(p));
    accessor.write(p, val);
  }
  inline FT *ptr(const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.ptr(p);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t strides[1],
                 size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    strides[0] = accessor.strides[0] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor[p];
  }

public:
  Realm::AffineAccessor<FT, 1, T> accessor;
  FieldID field;
  Rect<1, T> bounds;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = 1;
};

// Write-only FieldAccessor specialization
template <typename FT, int N, typename T, bool CB>
class FieldAccessor<LEGION_WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB> {
private:
  static_assert(N > 0, "DIM must be positive");

public:
  FieldAccessor(void) {}

public:
  inline void write(const Point<N, T> &p, FT val) const { accessor.write(p, val); }
  inline FT *ptr(const Point<N, T> &p) const { return accessor.ptr(p); }
  inline FT *ptr(const Rect<N, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t strides[N],
                 size_t field_size = sizeof(FT)) const
  {
    for(int i = 0; i < N; i++)
      strides[i] = accessor.strides[i] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<N, T> &p) const { return accessor[p]; }
  inline ArraySyntax::AffineSyntaxHelper<
      FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>, CB>,
      FT, N, T, 2, LEGION_WRITE_DISCARD>
  operator[](T index) const
  {
    return ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                      CB>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
  }

public:
  Realm::AffineAccessor<FT, N, T> accessor;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = N;
};

// Write-only FieldAccessor specialization
// with bounds checks
template <typename FT, int N, typename T>
class FieldAccessor<LEGION_WRITE_ONLY, FT, N, T, Realm::AffineAccessor<FT, N, T>, true> {
private:
  static_assert(N > 0, "DIM must be positive");

public:
  // No CUDA support due to PhysicalRegion constructor
  FieldAccessor(void) {}

public:
  inline void write(const Point<N, T> &p, FT val) const
  {
    assert(bounds.contains(p));
    accessor.write(p, val);
  }
  inline FT *ptr(const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.ptr(p);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<N, T> &r, size_t strides[N],
                 size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    for(int i = 0; i < N; i++)
      strides[i] = accessor.strides[i] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<N, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor[p];
  }
  inline ArraySyntax::AffineSyntaxHelper<
      FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                    true>,
      FT, N, T, 2, LEGION_WRITE_DISCARD>
  operator[](T index) const
  {
    return ArraySyntax::AffineSyntaxHelper<
        FieldAccessor<LEGION_WRITE_DISCARD, FT, N, T, Realm::AffineAccessor<FT, N, T>,
                      true>,
        FT, N, T, 2, LEGION_WRITE_DISCARD>(*this, Point<1, T>(index));
  }

public:
  Realm::AffineAccessor<FT, N, T> accessor;
  FieldID field;
  Rect<N, T> bounds;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = N;
};

// Write-only FieldAccessor specialization with
// N == 1 to avoid array ambiguity
template <typename FT, typename T, bool CB>
class FieldAccessor<LEGION_WRITE_ONLY, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, CB> {
public:
  FieldAccessor(void) {}

public:
  inline void write(const Point<1, T> &p, FT val) const { accessor.write(p, val); }
  inline FT *ptr(const Point<1, T> &p) const { return accessor.ptr(p); }
  inline FT *ptr(const Rect<1, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t strides[1],
                 size_t field_size = sizeof(FT)) const
  {
    strides[0] = accessor.strides[0] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<1, T> &p) const { return accessor[p]; }

public:
  Realm::AffineAccessor<FT, 1, T> accessor;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = 1;
};

// Write-only FieldAccessor specialization with
// N == 1 to avoid array ambiguity and bounds checks
template <typename FT, typename T>
class FieldAccessor<LEGION_WRITE_ONLY, FT, 1, T, Realm::AffineAccessor<FT, 1, T>, true> {
public:
  // No CUDA support due to PhysicalRegion constructor
  FieldAccessor(void) {}

public:
  inline void write(const Point<1, T> &p, FT val) const
  {
    assert(bounds.contains(p));
    accessor.write(p, val);
  }
  inline FT *ptr(const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor.ptr(p);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    assert(Internal::is_dense_layout(r, accessor.strides, field_size));
    return accessor.ptr(r.lo);
  }
  inline FT *ptr(const Rect<1, T> &r, size_t strides[1],
                 size_t field_size = sizeof(FT)) const
  {
    assert(bounds.contains_all(r));
    strides[0] = accessor.strides[0] / field_size;
    return accessor.ptr(r.lo);
  }
  inline FT &operator[](const Point<1, T> &p) const
  {
    assert(bounds.contains(p));
    return accessor[p];
  }

public:
  Realm::AffineAccessor<FT, 1, T> accessor;
  FieldID field;
  Rect<1, T> bounds;

public:
  typedef FT value_type;
  typedef FT &reference;
  typedef const FT &const_reference;
  static const int dim = 1;
};