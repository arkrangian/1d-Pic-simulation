import csv

def listToCsv(list:list, fileName: str):

    # Open a CSV file for writing
    with open(fileName, 'w', newline='') as csvfile:

        # Create a writer object for the CSV file
        writer = csv.writer(csvfile)

        # Write the data to the CSV file
        writer.writerows(list)
