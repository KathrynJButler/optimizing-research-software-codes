Usage
===
---

## Note
Refer to the RNX2DB manual documentation to download requirements. The documentation is a outdated so change the name of the folder to gns-python_main instead of rnx2db when you go through the document. 
Change the naviation, observation, output dir appropriately to your current directory.

This script will run faster on Linux environment.

## Configuration file
Settings can be changed by editing config.yaml file,

## Create compactdb.csv

```
$ python rinex.py
```

This script will read observation(.*o file) and navigation (.rnx) from data directory, parse and write compactdb.csv file
