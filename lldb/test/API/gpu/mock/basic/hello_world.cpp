#include <stdio.h>

struct ShlibInfo {
  const char *path;
  ShlibInfo *next;
};

ShlibInfo g_shlib_list = {"/tmp/a.out", nullptr};

int gpu_first_stop() { return puts(__FUNCTION__); }
int gpu_initialize() { return puts(__FUNCTION__); }
int gpu_shlib_load() { return puts(__FUNCTION__); }
int gpu_third_stop() { return puts(__FUNCTION__); }
int gpu_kernel() {
  // MOCK GPU BREAKPOINT
  return puts(__FUNCTION__);
}

int main(int argc, const char **argv) {
  gpu_first_stop();
  // CPU BREAKPOINT - BEFORE INITIALIZE
  gpu_initialize();
  // CPU BREAKPOINT - AFTER INITIALIZE
  gpu_shlib_load();
  gpu_third_stop();
  gpu_shlib_load();
  gpu_kernel();
  return 0;
}
