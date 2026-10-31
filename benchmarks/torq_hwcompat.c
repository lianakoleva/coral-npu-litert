/*
 * Torq hardware-compatibility ioctl shim.
 *
 * Some Astra kernel drivers do not implement the Torq hardware-compatibility
 * ioctl (0xc004530b).  The runtime logs
 *
 *   Warning: Hardware compatibility check not supported by kernel driver
 *   UNAVAILABLE; incompatible hardware, executable requires hw id 0
 *
 * and refuses to create the executable.  Prebuilt models all target hw id 0,
 * so answer the query with 0 and let the driver load the network.
 *
 * Build on the board:
 *   cc -shared -fPIC -O2 -o libtorq_hwcompat.so torq_hwcompat.c -ldl
 *
 * Use:
 *   LD_PRELOAD=/path/to/libtorq_hwcompat.so <torq program>
 */

#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <sys/ioctl.h>

#define TORQ_HW_COMPAT_IOC 0xc004530bUL

static int (*real_ioctl)(int, unsigned long, ...) = 0;

int ioctl(int fd, unsigned long request, ...) {
  va_list ap;
  void *arg;
  va_start(ap, request);
  arg = va_arg(ap, void *);
  va_end(ap);

  if (!real_ioctl) {
    real_ioctl = (int (*)(int, unsigned long, ...))dlsym(RTLD_NEXT, "ioctl");
  }
  if (request == TORQ_HW_COMPAT_IOC) {
    if (arg) {
      *(uint32_t *)arg = 0;
    }
    return 0;
  }
  return real_ioctl(fd, request, arg);
}
