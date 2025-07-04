import re
from collections import defaultdict

# Input and output file names
input_file = 'model_file.txt'
output_file = 'parameter_report.txt'

# Regex to extract information from each line
pattern = re.compile(r'(v\.[\w\.]+)\s+(F\d+)\s+\[([^\]]+)\]')

# Initialize counters
category_counts = defaultdict(lambda: defaultdict(int))
combined_categories = defaultdict(lambda: defaultdict(int))

def num_params(shape_str):
    """Calculate number of parameters from shape string."""
    dims = list(map(int, shape_str.split(',')))
    count = 1
    for d in dims:
        count *= d
    return count

# Read and process the file
with open(input_file, 'r') as f:
    for line in f:
        match = pattern.match(line.strip())
        if not match:
            continue

        name, dtype, shape_str = match.groups()
        param_count = num_params(shape_str)
        lower_name = name.lower()

        # Categorize parameter
        if ".attn_q." in lower_name:
            category = "query"
            combined_categories["attention_weights"][dtype] += param_count
        elif ".attn_k." in lower_name:
            category = "key"
            combined_categories["attention_weights"][dtype] += param_count
        elif ".attn_v." in lower_name:
            category = "value"
            combined_categories["attention_weights"][dtype] += param_count
        elif ".attn_output." in lower_name:
            category = "attn_output"
            combined_categories["attention_weights"][dtype] += param_count
        elif ".fc1." in lower_name:
            category = "fc1"
            combined_categories["fc_weights"][dtype] += param_count
        elif ".fc2." in lower_name:
            category = "fc2"
            combined_categories["fc_weights"][dtype] += param_count
        elif "layer_norm1" in lower_name:
            category = "layer_norm1"
        elif "layer_norm2" in lower_name:
            category = "layer_norm2"
        else:
            category = "other"
            combined_categories["other"][dtype] += param_count

        category_counts[category][dtype] += param_count

# Write to output report
with open(output_file, 'w') as report:
    report.write("📊 Parameter Type Summary (Individual Types)\n")
    report.write("=" * 50 + "\n")
    for category, dtype_dict in sorted(category_counts.items()):
        report.write(f"{category}:\n")
        for dtype, count in dtype_dict.items():
            report.write(f"  {dtype}: {count:,} parameters\n")
        report.write("\n")

    report.write("\n🔸 Combined Category Summary\n")
    report.write("=" * 50 + "\n")
    for category, dtype_dict in sorted(combined_categories.items()):
        report.write(f"{category}:\n")
        for dtype, count in dtype_dict.items():
            report.write(f"  {dtype}: {count:,} parameters\n")
        report.write("\n")

print(f"✅ Report generated in '{output_file}'")
