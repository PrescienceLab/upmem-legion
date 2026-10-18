/* Copyright 2024 Stanford University, Los Alamos National Laboratory,
 *                Northwestern University
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

// z = alpha * x + y with one DPU task per subregion; each DPU task runs on a
// whole UPMEM rank.

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/time.h>

#include <legion.h>
#include <common.h>

#if !defined(LEGION_USE_UPMEM)
#error Legion not compiled with UPMEM enabled
#endif

using namespace Legion;

enum TaskIDs
{
  TOP_LEVEL_TASK_ID,
  INIT_FIELD_TASK_ID,
  DAXPY_TASK_ID,
  SUM_TASK_ID,
  CHECK_TASK_ID,
};

enum FieldIDs
{
  FID_X,
  FID_Y,
  FID_Z,
};

typedef struct {
  TYPE alpha;
  Realm::Upmem::Kernel *kernel;
} DPU_TASK_ARGS;

#define DPU_LAUNCH_BINARY "dpu/daxpy_kernel.up.o"

static double now(void)
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec / 1e6;
}

void top_level_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                    Context ctx, Runtime *runtime)
{
  Realm::Upmem::Kernel *kern = new Realm::Upmem::Kernel(DPU_LAUNCH_BINARY);
  kern->load();

  long long num_elements = 1 << 20;
  int num_subregions = 0;
  int iterations = 1;
  {
    const InputArgs &command_args = Runtime::get_input_args();
    for(int i = 1; i < command_args.argc; i++) {
      if(!strcmp(command_args.argv[i], "-n"))
        num_elements = atoll(command_args.argv[++i]);
      if(!strcmp(command_args.argv[i], "-b"))
        num_subregions = atoi(command_args.argv[++i]);
      if(!strcmp(command_args.argv[i], "-i"))
        iterations = atoi(command_args.argv[++i]);
    }
  }
  if(num_subregions == 0) {
    Machine::ProcessorQuery q(Machine::get_machine());
    q.only_kind(Processor::DPU_PROC);
    num_subregions = q.count();
  }
  printf("daxpy: %lld elements, %d subregions, %d iterations\n", num_elements,
         num_subregions, iterations);

  Rect<1> elem_rect(0, num_elements - 1);
  IndexSpace is = runtime->create_index_space(ctx, elem_rect);
  FieldSpace fs = runtime->create_field_space(ctx);
  {
    FieldAllocator allocator = runtime->create_field_allocator(ctx, fs);
    allocator.allocate_field(sizeof(TYPE), FID_X);
    allocator.allocate_field(sizeof(TYPE), FID_Y);
    allocator.allocate_field(sizeof(TYPE), FID_Z);
  }
  LogicalRegion input_lr = runtime->create_logical_region(ctx, is, fs);
  LogicalRegion output_lr = runtime->create_logical_region(ctx, is, fs);

  Rect<1> color_bounds(0, num_subregions - 1);
  IndexSpace color_is = runtime->create_index_space(ctx, color_bounds);
  IndexPartition ip = runtime->create_equal_partition(ctx, is, color_is);
  LogicalPartition input_lp = runtime->get_logical_partition(ctx, input_lr, ip);
  LogicalPartition output_lp = runtime->get_logical_partition(ctx, output_lr, ip);

  ArgumentMap arg_map;
  IndexLauncher init_launcher(INIT_FIELD_TASK_ID, color_is, TaskArgument(NULL, 0),
                              arg_map);
  init_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0, WRITE_DISCARD, EXCLUSIVE, input_lr));
  init_launcher.region_requirements[0].add_field(FID_X);
  runtime->execute_index_space(ctx, init_launcher);
  init_launcher.region_requirements[0].privilege_fields.clear();
  init_launcher.region_requirements[0].instance_fields.clear();
  init_launcher.region_requirements[0].add_field(FID_Y);
  runtime->execute_index_space(ctx, init_launcher).wait_all_results();

  const TYPE alpha = rand() % 8192;
  DPU_TASK_ARGS args;
  args.alpha = alpha;
  args.kernel = kern;
  IndexLauncher daxpy_launcher(DAXPY_TASK_ID, color_is,
                               TaskArgument(&args, sizeof(args)), arg_map);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(input_lp, 0, READ_ONLY, EXCLUSIVE, input_lr));
  daxpy_launcher.region_requirements[0].add_field(FID_X);
  daxpy_launcher.region_requirements[0].add_field(FID_Y);
  daxpy_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0, WRITE_DISCARD, EXCLUSIVE, output_lr));
  daxpy_launcher.region_requirements[1].add_field(FID_Z);

  for(int it = 0; it < iterations; it++) {
    double start = now();
    runtime->execute_index_space(ctx, daxpy_launcher).wait_all_results();
    printf("iteration %d: daxpy (h2d + kernel) %.6f s\n", it, now() - start);
  }

  // sum(z): one partial per DPU, folded per rank by the runtime, then across
  //  ranks by Legion
  IndexLauncher sum_launcher(SUM_TASK_ID, color_is, TaskArgument(&args, sizeof(args)),
                             arg_map);
  sum_launcher.add_region_requirement(
      RegionRequirement(output_lp, 0, READ_ONLY, EXCLUSIVE, output_lr));
  sum_launcher.region_requirements[0].add_field(FID_Z);
  double sum_start = now();
  int64_t dpu_sum = runtime->execute_index_space(ctx, sum_launcher, LEGION_REDOP_SUM_INT64)
                        .get_result<int64_t>();
  printf("sum(z) on DPUs %.6f s\n", now() - sum_start);

  double start = now();
  TaskLauncher check_launcher(CHECK_TASK_ID, TaskArgument(&alpha, sizeof(alpha)));
  check_launcher.add_region_requirement(
      RegionRequirement(input_lr, READ_ONLY, EXCLUSIVE, input_lr));
  check_launcher.region_requirements[0].add_field(FID_X);
  check_launcher.region_requirements[0].add_field(FID_Y);
  check_launcher.add_region_requirement(
      RegionRequirement(output_lr, READ_ONLY, EXCLUSIVE, output_lr));
  check_launcher.region_requirements[1].add_field(FID_Z);
  int64_t host_sum = runtime->execute_task(ctx, check_launcher).get_result<int64_t>();
  printf("check (d2h + verify) %.6f s\n", now() - start);
  if(dpu_sum != host_sum) {
    printf("FAILURE! sum(z): DPUs %lld, host %lld\n", (long long)dpu_sum,
           (long long)host_sum);
    abort();
  }
  printf("sum(z) matches: %lld\n", (long long)dpu_sum);

  runtime->destroy_logical_region(ctx, input_lr);
  runtime->destroy_logical_region(ctx, output_lr);
  runtime->destroy_field_space(ctx, fs);
  runtime->destroy_index_space(ctx, is);
}

void init_field_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                     Context ctx, Runtime *runtime)
{
  FieldID fid = *(task->regions[0].privilege_fields.begin());
  const AccessorWD acc(regions[0], fid);
  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  unsigned seed = rect.lo[0] * 3 + fid;
  for(PointInRectIterator<1> pir(rect); pir(); pir++)
    acc[*pir] = rand_r(&seed) % 8192;
}

void daxpy_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                Context ctx, Runtime *runtime)
{
  DPU_TASK_ARGS task_args = *((DPU_TASK_ARGS *)task->args);

  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());

  DPU_LAUNCH_ARGS args;
  args.op = OP_DAXPY;
  args.alpha = task_args.alpha;
  args.rect = rect;
  args.acc_x = AccessorRO(regions[0], FID_X);
  args.acc_y = AccessorRO(regions[0], FID_Y);
  args.acc_z = AccessorWD(regions[1], FID_Z);
  task_args.kernel->launch((void **)&args, "ARGS", sizeof(DPU_LAUNCH_ARGS));
}

int64_t sum_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                 Context ctx, Runtime *runtime)
{
  DPU_TASK_ARGS task_args = *((DPU_TASK_ARGS *)task->args);

  DPU_LAUNCH_ARGS args;
  args.op = OP_SUM;
  args.rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  args.acc_x = AccessorRO(regions[0], FID_Z);
  return task_args.kernel->launch_reduce<SumReduction<int64_t> >(
      (void **)&args, "ARGS", sizeof(DPU_LAUNCH_ARGS), "PARTIAL");
}

int64_t check_task(const Task *task, const std::vector<PhysicalRegion> &regions,
                   Context ctx, Runtime *runtime)
{
  const TYPE alpha = *((const TYPE *)task->args);
  const AccessorRO acc_x(regions[0], FID_X);
  const AccessorRO acc_y(regions[0], FID_Y);
  const AccessorRO acc_z(regions[1], FID_Z);

  Rect<1> rect = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  size_t errors = 0;
  int64_t sum = 0;
  for(PointInRectIterator<1> pir(rect); pir(); pir++) {
    TYPE expected = alpha * acc_x[*pir] + acc_y[*pir];
    TYPE received = acc_z[*pir];
    sum += received;
    if(expected != received) {
      if(errors < 10)
        printf("mismatch at %lld: expected %d, received %d\n", (long long)(*pir)[0],
               (int)expected, (int)received);
      errors++;
    }
  }
  if(errors == 0) {
    printf("SUCCESS!\n");
  } else {
    printf("FAILURE! %zu errors\n", errors);
    abort();
  }
  return sum;
}

int main(int argc, char **argv)
{
  Runtime::set_top_level_task_id(TOP_LEVEL_TASK_ID);

  {
    TaskVariantRegistrar registrar(TOP_LEVEL_TASK_ID, "top_level");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    Runtime::preregister_task_variant<top_level_task>(registrar, "top_level");
  }
  {
    TaskVariantRegistrar registrar(INIT_FIELD_TASK_ID, "init_field");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<init_field_task>(registrar, "init_field");
  }
  {
    TaskVariantRegistrar registrar(DAXPY_TASK_ID, "daxpy");
    registrar.add_constraint(ProcessorConstraint(Processor::DPU_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<daxpy_task>(registrar, "daxpy");
  }
  {
    TaskVariantRegistrar registrar(SUM_TASK_ID, "sum");
    registrar.add_constraint(ProcessorConstraint(Processor::DPU_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int64_t, sum_task>(registrar, "sum");
  }
  {
    TaskVariantRegistrar registrar(CHECK_TASK_ID, "check");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<int64_t, check_task>(registrar, "check");
  }

  return Runtime::start(argc, argv);
}
