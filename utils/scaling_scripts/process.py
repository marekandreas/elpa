#! /usr/bin/env python3
import os
import subprocess

directories = ["out", "out_1", "out_1818"]
with open("results_raw.txt", "w") as logfile:
  for rootdir in directories:
    path = subprocess.check_output('pwd')[:-1].decode() + "/"
    for subdir, dirs, files in os.walk(rootdir):
      #  for file in files:
      #      print(os.path.join(subdir, file))
       # print(subdir, dirs, files )
        if(len(files) != 0):
            #print(subdir, dirs, files)
            print("cd " + path + subdir)
            logfile.write(subdir + "\n")
            os.chdir(path + subdir)

            method = subdir.split("/")[-1]
            with open("tab.txt", "w") as outfile:
              if(method == "elpa1"):
                parser = "parse_elpa1"
              elif(method == "elpa2"):
                parser = "parse_elpa2"
              else:
                parser = "parse_mkl"

              ps = subprocess.Popen(path + parser, stdout=subprocess.PIPE)
              output = subprocess.check_output(('column', '-t'), stdin = ps.stdout).decode()
              ps.wait()
              outfile.write(output)
              logfile.write(output)
              logfile.write("\n")


            os.chdir(path)
