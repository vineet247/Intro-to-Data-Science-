import MySQLdb
import os
import string

db = MySQLdb.connect (host="elk-2.mysql.database.azure.com",
    user="user@elk-2",
    passwd="Tn10z@6815",
    db="azuredb",
    local_infile = 1) #Grants permission to write to db from an input file. Without this you get sql Error: (1148, 'The used command is not allowed with this MySQL version')

print("\nConnection to DB established\n")

#The statement 'IGNORE 1 LINES' below makes the Python script ignore first line on csv file
#You can execute the sql below on the mysql bash to test if it works
#sqlLoadData = r"""load data local infile 'C:\Users\vinee\Desktop\Github\elk\datasets\fifa19' into table azuredb.fifa19 FIELDS TERMINATED BY ',' ENCLOSED BY '"' LINES TERMINATED BY '\n' IGNORE 1 LINES;"""


cursor = db.cursor()
Query = r""" LOAD DATA LOCAL INFILE 'C:\\Users\\vinee\\Desktop\\Github\\elk\\datasets\\fifa19\\data.csv'
INTO TABLE fifa19
FIELDS TERMINATED BY ','
ENCLOSED BY '"'
LINES TERMINATED BY '\n'
IGNORE 1 ROWS """

#Query = r""" LOAD DATA LOCAL INFILE 'data.csv'
#INTO TABLE fifa19
#FIELDS TERMINATED BY ','
#ENCLOSED BY '"'
#LINES TERMINATED BY '\n'
#IGNORE 1 ROWS """


cursor.execute(Query)
db.commit()
cursor.close()
