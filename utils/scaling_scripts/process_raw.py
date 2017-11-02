#! /usr/bin/env python
from collections import namedtuple
from subprocess import Popen, PIPE, STDOUT

EntryType = namedtuple("EntryType", "directory run dir_raw header lines")
with open("results_raw.txt", "r") as rawfile:
  expect = "directory"
  entries = []
  entry = []
  for line in rawfile:
    if(line == "\n"):
      if(expect == "entry"):
        entries.append(entry)
        expect = "directory"
      else:
        raise("cannot be")
    else:
      if(expect == "directory"):
        directory = line.split('/', 1)[0]
        run = line.split('/',1)[1][:-1].split('/')
        run[2] = int(run[2])
        run[3] = int(run[3])
        dir_raw = line[:-1]
        entry = EntryType(directory, run , dir_raw, [], [])
        expect = "heading"
        #print("next entry: " + line, end="")
      elif (expect == "heading"):
        entry.header.extend(line.split())
        expect = "entry"
        #print("heading: " + line, end="")
      elif (expect == "entry"):
        entry.lines.append(line.split())


entries.sort(key = lambda entry: entry.run)
entries.sort(key = lambda entry: entry.run[0], reverse=True)
print(entries)
for e in entries:
  print(e.directory, e.run)

with open("results_sorted.txt", "w") as sortedfile:
  lines_to_write = []
  for idx in range(len(entries)):
    e = entries[idx]
    for l in e.lines:
      lw = l[:-1]
      lw.append(e.dir_raw + "/" + l[-1])
      lines_to_write.append(lw)

    if(idx == len(entries) - 1 or e.run != entries[idx+1].run):
      if(e.run != entries[0].run):
        sortedfile.write("\n")
      sortedfile.write(" ".join(str(x) for x in e.run) + "\n")
      group_string = " ".join(str(x) for x in e.header) + "\n"
      lines_to_write.sort(key = lambda line: int(line[0]))
      for lw in lines_to_write:
        group_string += " ".join(lw) + "\n"

      p = Popen(['column', '-t'], stdout=PIPE, stdin=PIPE, stderr=PIPE)
      stdout_data = p.communicate(input=group_string.encode())[0]
      sortedfile.write(stdout_data.decode())

      lines_to_write = []


