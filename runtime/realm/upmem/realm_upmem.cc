#include <vector>
#include <realm.h>

#include "realm/upmem/realm_upmem.h"


using namespace Realm;


EXTERNC void *mem_alloc(unsigned int);

EXTERNC void *make_vector_of_int(int n)
{
  std::vector<int> *v = new std::vector<int>;
  v->resize(n);
  return (void*) v;
}

EXTERNC void  set_vector_of_int(void *v, int pos, int val)
{
  std::vector<int> *l = (std::vector<int>*)v;
  (*l)[pos] = val;
}
  
EXTERNC int   get_vector_of_int(void *v, int pos) {
  std::vector<int> *l = (std::vector<int>*)v;
  return (*l)[pos];
}
  
EXTERNC void *add_vectors_of_ints(void *left, void *right)
{
  std::vector<int> *res = new std::vector<int>;
  std::vector<int> *l = (std::vector<int>*)left;
  std::vector<int> *r = (std::vector<int>*)right;
  int n  = l->size();
  res->resize(n);
  
  for (auto i=0;i<n;i++) {
    (*res)[i] = (*l)[i] + (*r)[i];
  }
  
  return (void*)res;
}
  



EXTERNC void * malloc(unsigned int n);

void * operator new[](unsigned int n)
{
  return malloc(n);
}

void * operator new(unsigned int n)
{
  return malloc(n);
}


void operator delete(void *v) noexcept
{
  //free(v);
}

namespace std {
void __throw_length_error(char const*)
{}

void __throw_bad_alloc(void)
{}

void __throw_bad_array_new_length()
{}
}

