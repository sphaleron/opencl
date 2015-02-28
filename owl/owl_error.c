#include "owl_errno.h"

#include <stdio.h>

static const char* cl_error_to_string(int error);

// Make this static?
owl_error_handler_t* owl_error_handler = NULL;

void owl_error(const char* reason, const char* file, int line, int owl_errno) {
   if (owl_error_handler) {
      (*owl_error_handler)(reason, file, line, owl_errno);
      return;
   }

   // Default error handler

   if (reason == NULL && owl_errno < OWL_SUCCESS)
      reason = cl_error_to_string(owl_errno);

   fprintf(stderr, "%s:%d\n", file, line);
   if (reason != NULL)
      fprintf(stderr, "Error: %s\n", reason);
   fflush(stderr);

   abort();
}

// TODO
static const char* cl_error_to_string(int error) {
   return "foo";
}
