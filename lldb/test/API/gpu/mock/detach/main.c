// The mock GPU plug-in connects its GPU target when this is called, and sets
// a breakpoint on gpu_shlib_load.
void gpu_initialize(void) {}

// The mock GPU plug-in asks lldb to load the GPU target's libraries when this
// is called.
void gpu_shlib_load(void) {}

int main(void) {
  gpu_initialize();
  int result = 0; // break here
  gpu_shlib_load();
  return result; // break again
}
