// The mock GPU plug-in connects its GPU target when this is called, and lldb
// then loads the libraries that the plug-in reports for the GPU target.
void gpu_initialize(void) {}

int main(void) {
  gpu_initialize();
  return 0; // break here
}
