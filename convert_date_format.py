import csv
import datetime

# Open the input CSV file
with open("broken_dates_2012.csv", "r") as f_in:
    # Create a CSV reader
    reader = csv.reader(f_in)

    # Open the output CSV file
    with open("new_dates_dax.csv", "w", newline="") as f_out:
        # Create a CSV writer
        writer = csv.writer(f_out)

        # Iterate over the rows in the input file
        for row in reader:
            # Get the date from the first column
            date_str = row[0]

            # Parse the date string using the mm/dd/yyyy format
            date = datetime.datetime.strptime(date_str, "%d/%m/%Y")

            # Format the date using the dd/mm/yyyy format
            new_date_str = date.strftime("%m/%d/%Y")

            # Replace the date in the first column with the new date string
            row[0] = new_date_str

            # Write the modified row to the output file
            writer.writerow(row)
