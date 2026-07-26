import os
import sys

try:
    import unbalancedgw
except ImportError:
    print("Error: unbalancedgw is not installed in this python environment.")
    sys.exit(1)

# Find the path to the _vanilla_utils.py file inside the installed package
package_dir = os.path.dirname(unbalancedgw.__file__)
target_file = os.path.join(package_dir, '_vanilla_utils.py')

if not os.path.exists(target_file):
    print(f"Error: Could not find {target_file}")
    sys.exit(1)

with open(target_file, 'r') as f:
    content = f.read()

# The buggy line from the vanilla_ugw solver
buggy_line = "torch.ones([1]))"
fixed_line = "torch.ones([1], device=u.device))"

if buggy_line in content:
    content = content.replace(buggy_line, fixed_line)
    with open(target_file, 'w') as f:
        f.write(content)
    print(f"✅ Successfully patched {target_file} for GPU support!")
elif fixed_line in content:
    print(f"✅ The file {target_file} is already patched.")
else:
    print(f"⚠️ Could not find the exact line to patch in {target_file}. It may have been updated or modified differently.")
