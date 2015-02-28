/*
 * Elementary support for error handling, mostly copied from GSL. Handle both
 * OpenCL error codes and library specific errors.
 */

#ifndef OWL_ERRNO_H
#define OWL_ERRNO_H

#include <CL/cl.h>

// This is fairly fragile: we rely on OpenCL errors being negative (and success = 0)
// and that the compiler will give enums positive values (if the first one is 0)
// so that there are no conflicts. If this breaks down at some point, do something
// more complicated, like defining explicit values for all error codes.
enum {
   OWL_SUCCESS = CL_SUCCESS,
   OWL_EINVAL,
   OWL_NOMEM
};

void owl_error(const char* reason, const char* file, int line, int owl_errno);

typedef void owl_error_handler_t (const char* reason, const char* file,
                                  int line, int owl_errno);

// Macros for signaling an error. The reason can also be left NULL,
// in which case the name of the error code will be used.
#define OWL_ERROR(reason, owl_errno) \
   do { \
      owl_error(reason, __FILE__, __LINE__, owl_errno); \
      return owl_errno; \
   } while(0)

// For returning something that is not integer
#define OWL_ERROR_VAL(reason, owl_errno, value) \
   do { \
      owl_error(reason, __FILE__, __LINE__, owl_errno); \
      return value; \
   } while (0)

// OWL_ERROR_VOID: call the error handler, and then return
#define OWL_ERROR_VOID(reason, owl_errno) \
   do { \
      owl_error(reason, __FILE__, __LINE__, owl_errno); \
      return; \
   } while (0)

// OWL_ERROR_NULL suitable for out-of-memory conditions
#define OWL_ERROR_NULL(reason, owl_errno) OWL_ERROR_VAL(reason, owl_errno, 0)


#endif
