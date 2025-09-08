#!/usr/bin/env python3
"""
Minimal test of the end-to-end pipeline to verify everything works.
This runs with reduced iterations to complete quickly.
"""

import os
import sys
import subprocess

# Run the main script with minimal iterations
env = os.environ.copy()
env['TINYSTORIES_TEST_MODE'] = '1'

# Modify the main script temporarily to run minimal iterations
script_path = "train_tinystories_end2end.py"

# Read the original script
with open(script_path, 'r') as f:
    original_content = f.read()

# Create a test version with minimal iterations
test_content = original_content.replace(
    '"total_tokens": 327_680_000,',
    '"total_tokens": 10_000,  # TEST MODE'
).replace(
    '"eval_interval": 100,',
    '"eval_interval": 10,  # TEST MODE'
).replace(
    '"save_interval": 1000,',
    '"save_interval": 50,  # TEST MODE'
)

# Write test version
test_script = "test_tinystories_minimal_run.py"
with open(test_script, 'w') as f:
    f.write(test_content)

print("Running minimal test of the pipeline...")
print("This will:")
print("1. Download a small portion of data (if needed)")
print("2. Train tokenizer on small data") 
print("3. Tokenize the data")
print("4. Train model for just a few iterations")
print("5. Generate a sample")
print("\n" + "="*60 + "\n")

try:
    # Run the test script
    result = subprocess.run([sys.executable, test_script], check=True)
    print("\n✅ Test completed successfully!")
except subprocess.CalledProcessError as e:
    print(f"\n❌ Test failed with error: {e}")
finally:
    # Clean up test script
    if os.path.exists(test_script):
        os.unlink(test_script)
