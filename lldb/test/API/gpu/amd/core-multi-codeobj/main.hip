// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#include <hip/hip_runtime.h>
#include <cstdio>

void launch_alpha(int *d);
void launch_beta(int *d, int n);

int main() {
  int *da, *db;
  hipMalloc(&da, 32 * sizeof(int));
  hipMalloc(&db, 32 * sizeof(int));
  launch_alpha(da); // loads code object A
  hipDeviceSynchronize();
  launch_beta(db, 32); // loads code object B; breakpoint fires inside it
  hipDeviceSynchronize();
  printf("done\n");
  return 0;
}
