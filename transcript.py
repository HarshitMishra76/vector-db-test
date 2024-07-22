import re


# Function to clean the text
def clean_text(text):
    # Remove timestamps
    text_no_timestamps = re.sub(r'\d{2}:\d{2}:\d{2}\.\d{3}', '', text)
    # Remove line breaks
    text_no_line_breaks = ' '.join(text_no_timestamps.splitlines())
    # Remove extra spaces
    text_cleaned = re.sub(r'\s+', ' ', text_no_line_breaks).strip()
    return text_cleaned


# Read from input file
with open('LLMs in action: Strategies for crafting solutions.txt', 'r') as file:
    input_text = file.read()
    filename = file.name

# Clean the text
cleaned_text = clean_text(input_text)

# Write to output file
with open(f'references/{filename}', 'w') as file:
    file.write(cleaned_text)

print("Text has been cleaned and written to 'output.txt'")
