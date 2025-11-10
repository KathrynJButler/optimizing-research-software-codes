For processing a single day’s rinex observation and navigation file, set the following headers:

rinex:

multiday:  # options for multiday process
- enabled: false  # if true, it will read obs_location directory and try to process all data

singlefile:
- date: '2024-07-10'
- observation:
  - rnx2db_testdata/one_day/QRTR1920.24O
- navigation:
   - rnx2db_testdata/one_day/QRTR1920.24P
