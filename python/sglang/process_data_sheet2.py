import re
import argparse
import csv

def process_benchmark_file(input_file_path, output_file_path):
    # Read the content from the input file
    with open(input_file_path, 'r') as file:
        data = file.read()

    # Extract the data following the "Benchmark ..." line if present
    if "Benchmark ..." in data:
        benchmark_data = data.split("Benchmark ...")[1].strip()
    else:
        benchmark_data = data

    # Regex pattern to match batch_size and corresponding latencies
    batch_size_pattern = r"batch_size: (\d+),.*?Prefill\. latency:\s+([0-9\.]+) s"
    batch_size_pattern = r"batch_size:\s+(\d+).+?Decode\.\s+median\s+latency:\s+([0-9\.]+)\s+s"

    # Find all matches for batch_size and Prefill latencies
    batch_size_matches = re.findall(batch_size_pattern, benchmark_data, re.DOTALL)

    # Dictionary to hold batch_size and list of latencies
    batch_size_dict = {}

    for match in batch_size_matches:
        batch_size = match[0]
        latencies = match[1].split(",")

        if batch_size not in batch_size_dict:
            batch_size_dict[batch_size] = []

        batch_size_dict[batch_size].extend(lat.strip() for lat in latencies)

    # Write the data to CSV file
    with open(output_file_path, 'w', newline='') as out_file:
        writer = csv.writer(out_file)

        # Determine the maximum number of latencies for any batch size to create headers
        max_columns = max(len(latencies) for latencies in batch_size_dict.values())
        headers = ["batch_size"] + [f"latency_{i+1}" for i in range(max_columns)]
        writer.writerow(headers)

        # Write the data rows
        for batch_size, latencies in batch_size_dict.items():
            row = [batch_size] + latencies
            writer.writerow(row)

    print(f"Results written to: {output_file_path}")

def main():
    # Command line argument parser
    parser = argparse.ArgumentParser(description="Process and format Benchmark output for Excel")
    
    # Add arguments for input and output file paths
    parser.add_argument("input_file", help="Path to the input file")
    parser.add_argument("output_file", help="Path to the output file")
    
    # Parse the command line arguments
    args = parser.parse_args()
    
    # Call the processing function
    process_benchmark_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
