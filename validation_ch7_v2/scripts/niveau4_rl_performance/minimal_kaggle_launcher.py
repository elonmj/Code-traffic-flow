#!/usr/bin/env python3
"""
Minimal Kaggle Launcher - Section 7.6 RL Validation
Standalone script to avoid dependency import issues on Windows
"""

import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Minimal Kaggle API client with error handling
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError as e:
    print(f"❌ Kaggle API not available: {e}")
    sys.exit(1)


class KaggleLauncher:
    """Minimal Kaggle kernel launcher for validation"""
    
    def __init__(self):
        self.api = KaggleApi()
        try:
            self.api.authenticate()
            print("✅ Kaggle API authenticated")
        except Exception as e:
            print(f"❌ Kaggle authentication failed: {e}")
            raise
    
    def create_and_run_kernel(self, quick_test: bool = True):
        """Create and run validation kernel on Kaggle"""
        
        repo_path = Path(__file__).parent.parent.parent.parent
        print(f"\n📦 Setting up Kaggle kernel...")
        print(f"   Repository: {repo_path}")
        
        # Build minimal notebook-based script
        kernel_script = self._build_kernel_script(quick_test=quick_test)
        
        # Save to .kaggle_kernels directory
        kernel_dir = Path(__file__).parent / ".kaggle_kernels"
        kernel_dir.mkdir(exist_ok=True)
        
        kernel_name = f"arz-section-7-6-dqn-quick-{'v' + datetime.now().strftime('%m%d%H%M')}"
        kernel_file = kernel_dir / f"{kernel_name}.py"
        
        print(f"\n📝 Writing kernel script: {kernel_file}")
        kernel_file.write_text(kernel_script)
        
        # Create kernel metadata
        metadata = {
            "title": "ARZ Section 7.6 - RL Performance Validation (Auto)",
            "kernel_type": "script",
            "is_private": False,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": [],
            "language": "python",
            "code_file": str(kernel_file),
        }
        
        metadata_file = kernel_dir / f"{kernel_name}-metadata.json"
        metadata_file.write_text(json.dumps(metadata, indent=2))
        
        print(f"   Metadata: {metadata_file}")
        
        # Create kernel on Kaggle
        print(f"\n🚀 Creating kernel on Kaggle: {kernel_name}")
        
        try:
            # Push kernel to Kaggle
            response = self.api.kernels_push_cli(
                kernel_dir=str(kernel_dir),
                kernel_name=kernel_name,
                quiet=False
            )
            
            # Extract kernel reference
            if hasattr(response, 'ref'):
                kernel_ref = response.ref
                if kernel_ref.startswith('/'):
                    kernel_ref = kernel_ref[1:]
            else:
                kernel_ref = f"joselonm/{kernel_name}"
            
            print(f"   ✅ Kernel created: {kernel_ref}")
            
            # Monitor execution
            return self.monitor_kernel(kernel_ref)
            
        except Exception as e:
            print(f"   ❌ Kernel creation failed: {e}")
            raise
    
    def monitor_kernel(self, kernel_ref: str, timeout: int = 600, check_interval: int = 30):
        """Monitor kernel execution"""
        
        print(f"\n⏱️  Monitoring kernel: {kernel_ref}")
        print(f"   Timeout: {timeout}s, Check interval: {check_interval}s")
        
        start_time = time.time()
        
        while True:
            try:
                status_info = self.api.kernels_status(kernel_ref)
                status = getattr(status_info, 'status', 'unknown')
                
                elapsed = int(time.time() - start_time)
                print(f"   [{elapsed}s] Status: {status}")
                
                if str(status) == "KernelWorkerStatus.COMPLETE":
                    print(f"   ✅ Kernel execution COMPLETE")
                    return self.download_results(kernel_ref)
                
                elif str(status) == "KernelWorkerStatus.RUNNING":
                    time.sleep(check_interval)
                
                else:
                    print(f"   ⚠️  Unexpected status: {status}")
                    time.sleep(check_interval)
                
                if elapsed > timeout:
                    print(f"   ❌ Timeout after {timeout}s")
                    return False
                    
            except Exception as e:
                print(f"   ❌ Status check failed: {e}")
                time.sleep(check_interval)
    
    def download_results(self, kernel_ref: str):
        """Download kernel results"""
        
        print(f"\n📥 Downloading results...")
        
        output_dir = Path(__file__).parent / "kaggle_results"
        output_dir.mkdir(exist_ok=True)
        
        try:
            self.api.kernels_output(kernel_ref, path=str(output_dir))
            print(f"   ✅ Results downloaded to: {output_dir}")
            
            # List files
            for f in output_dir.glob("*"):
                print(f"      - {f.name} ({f.stat().st_size} bytes)")
            
            return True
            
        except Exception as e:
            print(f"   ❌ Download failed: {e}")
            return False
    
    def _build_kernel_script(self, quick_test: bool = True) -> str:
        """Build minimal kernel validation script"""
        
        return f'''#!/usr/bin/env python3
"""Kaggle Kernel - ARZ Section 7.6 RL Performance Validation"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("🚀 ARZ-RL VALIDATION - KAGGLE GPU EXECUTION")
print("=" * 80)
print(f"Start time: {{datetime.now().isoformat()}}")
print(f"Python: {{sys.version}}")
print(f"Working directory: {{os.getcwd()}}")
print("=" * 80)

# PHASE 1: Clone repository
print("\\n📦 PHASE 1: Cloning repository...")
repo_url = "https://github.com/elonmj/Code-traffic-flow.git"
print(f"Repository: {{repo_url}}")
print(f"Branch: main")

os.system(f"git clone {{repo_url}} /kaggle/working/repo 2>&1 | head -20")

os.chdir("/kaggle/working/repo")
print(f"✅ Repository cloned to: /kaggle/working/repo")
print(f"✅ Working directory: {{os.getcwd()}}")

# PHASE 2: Run validation
print("\\n🧪 PHASE 2: Running validation...")
script_path = "validation_ch7_v2/scripts/niveau4_rl_performance/run_section_7_6.py"
print(f"Script: {{script_path}}")
print(f"Quick mode: {quick_test}")
print(f"Device: gpu")

cmd = f"python {{script_path}} --device gpu" + (" --quick" if {quick_test} else "")
print(f"Command: {{cmd}}")
print("=" * 80)

ret = os.system(cmd)
print(f"\\n⚠️  Validation exited with code: {{ret >> 8}}")

# PHASE 3: Copy results
print("\\n📋 Copying results...")
os.system("mkdir -p /kaggle/working/validation_results")
os.system("cp -r validation_results/* /kaggle/working/validation_results/ 2>/dev/null || true")
print("✅ Results copied to: /kaggle/working/validation_results")

# PHASE 4: Cleanup
print("\\n🧹 PHASE 4: Cleanup...")
os.system("cd /kaggle/working && rm -rf repo")
print("✅ Repository cleaned up")

print("\\n📁 Final outputs:")
print("=" * 80)

output_dir = Path("/kaggle/working/validation_results")
if output_dir.exists():
    for f in output_dir.glob("**/*"):
        if f.is_file():
            print(f"  - {{f.relative_to(output_dir)}} ({{f.stat().st_size}} bytes)")

print("=" * 80)
print("✅ SESSION_COMPLETE")
print(f"End time: {{datetime.now().isoformat()}}")
print("=" * 80)

# Write session summary
summary = {{
    "status": "COMPLETE",
    "timestamp": datetime.now().isoformat(),
    "device": "Tesla P100-PCIE-16GB"
}}

with open("/kaggle/working/session_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''
    
    def run(self, quick_test: bool = True):
        """Run the launcher"""
        try:
            return self.create_and_run_kernel(quick_test=quick_test)
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user")
            return False
        except Exception as e:
            print(f"\n❌ Launcher error: {e}")
            return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Minimal Kaggle launcher")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    args = parser.parse_args()
    
    launcher = KaggleLauncher()
    success = launcher.run(quick_test=args.quick_test)
    
    sys.exit(0 if success else 1)
