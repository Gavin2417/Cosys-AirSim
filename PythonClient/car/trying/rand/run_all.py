import subprocess
from tqdm import tqdm

# Loop through test_id from 0 to 7
for test_id in tqdm(range(8), desc="Running tests"):
    try:
        # Run the script with the current test_id
        subprocess.run(
            ['python', 'test_SemanticKITTI.py', '--test_id', str(test_id)],
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"Test with test_id={test_id} failed with error: {e}")
