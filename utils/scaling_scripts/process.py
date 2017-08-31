#! /usr/bin/env python
import os
import subprocess

rootdir = "results"
path = subprocess.check_output('pwd')[:-1] + "/"

for subdir, dirs, files in os.walk(rootdir):
  #  for file in files:
  #      print os.path.join(subdir, file)
   # print subdir, dirs, files 
    if(len(files) != 0):
        #print subdir, dirs, files 
        #print("cd " + path + subdir)
        os.chdir(path + subdir)

        method = subdir.split("/")[-1]
        with open("tab.txt", "w") as outfile:
            if(method == "elpa1"):
                subprocess.call("parse_elpa1", stdout=outfile) 
            elif(method == "elpa2"):
                subprocess.call("parse_elpa2", stdout=outfile)
            else:
                subprocess.call("parse_mkl", stdout=outfile)

        os.chdir(path)
