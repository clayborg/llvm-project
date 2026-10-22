#include <unistd.h>

// The mock GPU plug-in sets a breakpoint here, and halts this process once it
// runs again, as a GPU plug-in does when its GPU stops.
void gpu_halt_native_process(void) {}

volatile int g_wait_for_the_halt = 1;

void wait_for_the_halt(void) {
  gpu_halt_native_process();
  for (int i = 0; i < 100 && g_wait_for_the_halt; ++i)
    usleep(100 * 1000);
}

int main(void) {
  wait_for_the_halt(); // step over this line
  return 0;
}
