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
int gpu_resume_and_wait_for_resume() { return puts(__FUNCTION__); }
int gpu_wait_for_stop() { return puts(__FUNCTION__); }

int main(int argc, const char **argv) {
  gpu_first_stop();
  printf("Starting mock GPU test...\n"); // CPU BREAKPOINT
  gpu_initialize();
  gpu_shlib_load();
  gpu_third_stop();
  gpu_shlib_load();
  gpu_resume_and_wait_for_resume();
  gpu_wait_for_stop();
  gpu_shlib_load();
  printf("Done.\n"); // CPU BREAKPOINT - AFTER
  return 0;
}
