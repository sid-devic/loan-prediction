import matplotlib
import csv

# Loan sub_grade rating: [9]
# Zip: [22] 
# employer length: [11], strip all but int or 10+
# income: [13]
# previous loans past 30 days due (Bad): [25] convert to int 
# 

with open('sample.csv') as csvRead:
    readCSV = csv.reader(csvRead, delimiter=',')
    
    for row in csvRead:
        print(row[16])
    '''
    with open('data.csv','w') as csvWrite:
        writeCSV = csv.writer(csvWrite) 
        for row in readCSV:
            readRow = [row[11], row[13], row[25], row[9]]
            writeCSV.writerow(readRow)
    '''
