#include <realm/upmem/realm_c_upmem.h>
// #include <realm/inst_layout.h>

#include <vector>

// using namespace Realm;

EXTERNC void* mem_alloc(unsigned int);

EXTERNC void* malloc(size_t n)
{
    return mem_alloc(n);
}

EXTERNC int access_point(int x, int y)
{
    return x;
    // RegionInstance linear_instance;
    // AffineAccessor<float, 2> linear_accessor(linear_instance, 0);

    // float value = linear_accessor[Point<2>(x, y)];
    // return value;
}

EXTERNC void* make_vector_of_int(int n)
{
    std::vector<int>* v = new std::vector<int>;
    v->resize(n);
    return (void*)v;
}

EXTERNC void set_vector_of_int(void* v, int pos, int val)
{
    std::vector<int>* l = (std::vector<int>*)v;
    (*l)[pos] = val;
}

EXTERNC int get_vector_of_int(void* v, int pos)
{
    std::vector<int>* l = (std::vector<int>*)v;
    return (*l)[pos];
}

EXTERNC void* add_vectors_of_ints(void* left, void* right)
{
    std::vector<int>* res = new std::vector<int>;
    std::vector<int>* l = (std::vector<int>*)left;
    std::vector<int>* r = (std::vector<int>*)right;
    int n = l->size();
    res->resize(n);

    for (auto i = 0; i < n; i++) {
        (*res)[i] = (*l)[i] + (*r)[i];
    }

    return (void*)res;
}

// void * operator new[](size_t n)
// {
//   return malloc(n);
// }

// void * operator new(size_t n)
// {
//   return malloc(n);
// }

// void operator delete[](void *v) noexcept
// {
//   //free(v);
// }

// void operator delete(void *v) noexcept
// {
//   //free(v);
// }

// namespace std {

// void __cxa_atexit();

// void __throw_length_error(char const*)
// {}

// void __throw_bad_alloc(void)
// {}

// void __throw_bad_array_new_length()
// {}
// }