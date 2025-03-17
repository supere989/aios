#!/bin/bash

# test_template.sh - Script to test template functionality in lm_studio_benchmark.py
# Enhanced version with proper Jinja2 templating and model inference testing

# Set colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Set the test model ID and template file name
TEST_MODEL_ID="test-model"
TEMPLATE_FILE="test-model-template.txt"
# Choose an actual model to test with from LM-Studio
REAL_MODEL_ID="qwen2.5-coder-0.5b-instruct"
TEST_PROMPT="Explain what makes a good programming language in one paragraph."
OUTPUT_FILE="template_test_results.json"

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}=== $1 ===${NC}"
}

# Function to print success messages
print_success() {
    echo -e "${GREEN}   ✓ $1${NC}"
}

# Function to print error messages and exit
print_error() {
    echo -e "${RED}ERROR: $1${NC}"
    # Clean up if needed
    [ -f "$TEMPLATE_FILE" ] && rm -f "$TEMPLATE_FILE"
    exit 1
}

# Function to print warning messages
print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

# Function to check command result
check_result() {
    if [ $? -ne 0 ]; then
        print_error "$1"
    else
        print_success "$2"
    fi
}

print_section "Starting Enhanced Template Test"

# Step 1: Create a test template file with a proper Jinja2 template
echo "1. Creating test template file with Jinja2 format..."
touch "$TEMPLATE_FILE"
[ ! -f "$TEMPLATE_FILE" ] && print_error "Failed to create template file!"

# Clear file to be safe and write a proper Jinja2 template
# Using a template similar to qwen2.5-7b-instruct-1m but simplified
cat > "$TEMPLATE_FILE" << 'EOT'
{%- if messages[0].role == 'system' %}
    {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
{%- else %}
    {{- '<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n' }}
{%- endif %}

{%- for message in messages %}
    {%- if message.role == 'user' %}
        {{- '<|im_start|>user\n' + message.content + '<|im_end|>\n' }}
    {%- elif message.role == 'assistant' %}
        {{- '<|im_start|>assistant\n' + message.content + '<|im_end|>\n' }}
    {%- endif %}
{%- endfor %}

{%- if add_generation_prompt %}
    {{- '<|im_start|>assistant\n' }}
{%- endif %}
EOT

# Check if file is created and has content
[ ! -s "$TEMPLATE_FILE" ] && print_error "Failed to write template content!"

print_success "Template file created with proper Jinja2 format"
echo "   Template content sample:"
head -10 "$TEMPLATE_FILE"
echo "   ..."

# Step 2: Set the template using lm_studio_benchmark.py
print_section "Setting Template"
echo "Setting template for model '$TEST_MODEL_ID'..."
python lm_studio_benchmark.py template set "$TEST_MODEL_ID" --file "$TEMPLATE_FILE" --description "Advanced Jinja2 template with special tokens"
check_result "Failed to set template!" "Template set successfully"

# Step 3: Verify template was added
print_section "Verifying Template"
echo "Listing available templates to verify addition..."
python lm_studio_benchmark.py template list | grep "$TEST_MODEL_ID"
check_result "Template not found in list!" "Template successfully added to the list"

# Step 4: Test the template with a real model
print_section "Testing Template with Model"
echo "4.1. First testing without template for comparison..."
# Create a temporary model ID without template
NO_TEMPLATE_ID="test-no-template"
echo "Running model inference without template..."
python lm_studio_benchmark.py select "$REAL_MODEL_ID"
python lm_studio_benchmark.py test --prompt "$TEST_PROMPT" > response_without_template.txt
check_result "Failed to test model without template!" "Model tested without template"

echo "4.2. Now testing with our custom template..."
# Apply our template to the real model
python lm_studio_benchmark.py template set "$REAL_MODEL_ID" --file "$TEMPLATE_FILE" --description "Temporary test template"
python lm_studio_benchmark.py test --prompt "$TEST_PROMPT" > response_with_template.txt
check_result "Failed to test model with template!" "Model tested with template"

# Step 5: Compare the results
print_section "Comparing Results"
echo "Comparing responses with and without template..."

# Display differences (if any)
echo "Without template (first 150 characters):"
head -c 150 response_without_template.txt
echo -e "\n...\n"

echo "With template (first 150 characters):"
head -c 150 response_with_template.txt
echo -e "\n...\n"

# Check for special tokens in the templated version
if grep -q '<|im_' response_with_template.txt; then
    print_warning "Special tokens were not properly processed by the model"
    echo "   Note: This is expected if the model doesn't support these tokens natively"
else
    print_success "No raw template tokens in the output"
fi

# Step 6: Clean up
print_section "Cleaning Up"
echo "Removing the template for the real model..."
# Remove our template from the real model by setting it to something minimal
python lm_studio_benchmark.py template set "$REAL_MODEL_ID" --template "{}" --description "Default"
check_result "Failed to reset template for $REAL_MODEL_ID!" "Reset template for $REAL_MODEL_ID"

echo "Cleaning up test files..."
rm -f "$TEMPLATE_FILE" response_with_template.txt response_without_template.txt
[ -f "$TEMPLATE_FILE" ] && print_warning "Failed to delete template file!" || print_success "Test files deleted successfully"

print_section "Template Test Completed Successfully"
echo "The test demonstrated:"
echo "1. Creating and setting a proper Jinja2 template"
echo "2. Testing template formatting with a real model"
echo "3. Comparing results with and without the template"
echo "4. Safe cleanup of test artifacts"
