#include <stdio.h>

struct ShlibInfo {
  const char *path;
  ShlibInfo *next;
};

ShlibInfo g_shlib_list = {"/tmp/a.out", nullptr};

int gpu_initialize() { return puts(__FUNCTION__); }
int gpu_shlib_load() { return puts(__FUNCTION__); }
int gpu_third_stop() { return puts(__FUNCTION__); }

int main(int argc, const char **argv) {
  gpu_initialize();
  gpu_shlib_load();
  gpu_third_stop();
  gpu_shlib_load();
  return 0; // BREAKPOINT
}
