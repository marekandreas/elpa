//    This file is part of ELPA.
//
//    The ELPA library was originally created by the ELPA consortium,
//    consisting of the following organizations:
//
//    - Rechenzentrum Garching der Max-Planck-Gesellschaft (RZG),
//    - Bergische Universität Wuppertal, Lehrstuhl für angewandte
//      Informatik,
//    - Technische Universität München, Lehrstuhl für Informatik mit
//      Schwerpunkt Wissenschaftliches Rechnen ,
//    - Fritz-Haber-Institut, Berlin, Abt. Theorie,
//    - Max-Plack-Institut für Mathematik in den Naturwissenschaften,
//      Leipzig, Abt. Komplexe Strukutren in Biologie und Kognition,
//      and
//    - IBM Deutschland GmbH
//
//
//    More information can be found here:
//    http://elpa.rzg.mpg.de/
//
//    ELPA is free software: you can redistribute it and/or modify
//    it under the terms of the version 3 of the license of the
//    GNU Lesser General Public License as published by the Free
//    Software Foundation.
//
//    ELPA is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.
//
//    You should have received a copy of the GNU Lesser General Public License
//    along with ELPA.  If not, see <http://www.gnu.org/licenses/>
//
//    ELPA reflects a substantial effort on the part of the original
//    ELPA consortium, and we ask you to respect the spirit of the
//    license that we chose: i.e., please contribute any changes you
//    may have back to the original ELPA library distribution, and keep
//    any derivatives of ELPA under the same license that we chose for
//    the original distribution, the GNU Lesser General Public License.
//
//
// --------------------------------------------------------------------------------------------------
#include <stdio.h>
#include <fcntl.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

#define NAME_LENGTH 4096
#define FILENAME "./mpi_stdout/std%3s_rank%04d.txt"

FILE *tout, *terr;
void dup_filename(char *filename, int dupfd);
void dup_fd(int fd, int dupfd);

int _mkdirifnotexists(const char *dir) {
    struct stat s;
    if (stat(dir, &s) != 0) {
        if (errno == ENOENT) {
            if (mkdir(dir, 0755) != 0) {
                perror("mkdir");
                return 0;
            } else {
                return 1;
            }
        } else {
            perror("stat()");
	    return 0;
        }
    } else if (!S_ISDIR(s.st_mode)) {
        fprintf(stderr, "\"%s\" does exist and is not a directory\n", dir);
        return 0;
    } else {
        return 1;
    }
}

int create_directories(void) {
    if (!_mkdirifnotexists("mpi_stdout")) return 0;
    return 1;
}

void redirect_stdout(int *myproc) {
  char buf[NAME_LENGTH];

  if (*myproc == 0) {
    snprintf(buf, NAME_LENGTH, "tee " FILENAME, "out", *myproc);
    tout = popen(buf, "w");
    dup_fd(fileno(tout), 1);

    snprintf(buf, NAME_LENGTH, "tee " FILENAME, "err", *myproc);
    terr = popen(buf, "w");
    dup_fd(fileno(terr), 2);
  } else {
    snprintf(buf, NAME_LENGTH, FILENAME, "out", *myproc);
    dup_filename(buf, 1);

    snprintf(buf, NAME_LENGTH, FILENAME, "err", *myproc);
    dup_filename(buf, 2);
  }

  return;
}

/* Redirect file descriptor dupfd to file filename */
void dup_filename(char *filename, int dupfd) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  if(fd < 0) {
    perror("open()");
    exit(1);
  }
  dup_fd(fd, dupfd);
}

/* Redirect file descriptor dupfd to file descriptor fd */
void dup_fd(int fd, int dupfd) {
  if(dup2(fd,dupfd) < 0) {
    perror("dup2()");
    exit(1);
  }
}
