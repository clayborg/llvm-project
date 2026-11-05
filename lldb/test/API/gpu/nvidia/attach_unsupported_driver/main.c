#include "attach.h"

#include <stdio.h>
#include <unistd.h>

int libcuda_stand_in(void);

// Usage: a.out READY
//
// Create READY once libcuda is loaded, so that a debugger can attach. The test
// kills the program. The time limit only stops a program that it leaves behind
// from running forever.
int main(int argc, char **argv) {
  lldb_enable_attach();
  // The call keeps the linker from dropping libcuda as unneeded.
  if (argc < 2 || libcuda_stand_in() != 0)
    return 1;

  FILE *ready = fopen(argv[1], "w");
  if (!ready)
    return 1;
  fclose(ready);

  sleep(600);
  return 0;
}
