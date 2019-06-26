#include "config.h"
#include "elpa/elpa_build_config.h"
#include <stdio.h>


/*
!f>#ifdef STORE_BUILD_CONFIG
!f> interface
!f>   subroutine print_build_config() &
!f>              bind(C, name="print_build_config")
!f>        use, intrinsic :: iso_c_binding
!f>   end subroutine
!f> end interface
!f>#endif
*/

void print_build_config(){
#ifdef STORE_BUILD_CONFIG
  printf("===============================================================\n");
  printf("    Output of the autoconf config.log created at build time    \n\n");
  printf(" In case of troubles with the ELPA library, please send the \n follwing output together with a problem description \n at elpa-library@mpcdf.mpg.de \n\n");
  printf("%s \n",elpa_build_object);
  printf("===============================================================\n");
#endif
}
