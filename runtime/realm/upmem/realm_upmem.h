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


#ifdef __cplusplus
#define EXTERNC extern "C"
#else
#define EXTERNC
#endif


#define __GLIBC_PREREQ(x, y) 0
#define malloc(x) mem_alloc(x)



EXTERNC void *make_vector_of_int(int n);
EXTERNC void  set_vector_of_int(void *v, int pos, int val);
EXTERNC int   get_vector_of_int(void *v, int pos);
EXTERNC void *add_vectors_of_ints(void *left, void *right);

#endif