#include <stdio.h>
#include <unistd.h>

int libcuda_stand_in(void);

int main(int argc, char **argv) {
  // The call keeps the linker from dropping libcuda as unneeded.
  if (libcuda_stand_in() != 0)
    return 1;

  // Tell the test that libcuda is loaded and it can attach.
  if (argc > 1) {
    FILE *marker = fopen(argv[1], "w");
    if (!marker)
      return 1;
    fclose(marker);
  }

  // The test kills this process during teardown; the time limit only stops an
  // orphan from lingering if that never happens.
  for (int i = 0; i < 600; ++i)
    sleep(1);
  return 0;
}
