"""
Test script to verify that all dependencies are properly installed
Run this after installing requirements.txt
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """Test if a module can be imported"""
    try:
        if package_name:
            importlib.import_module(package_name)
        else:
            importlib.import_module(module_name)
        print(f"✓ {module_name:20s} ... OK")
        return True
    except ImportError as e:
        print(f"✗ {module_name:20s} ... FAILED")
        print(f"  Error: {str(e)}")
        return False

def main():
    print("\n" + "="*60)
    print("Testing Plant Leaf Disease Detection Dependencies")
    print("="*60 + "\n")
    
    modules = [
        ("opencv-python", "cv2"),
        ("numpy", "numpy"),
        ("scikit-learn", "sklearn"),
        ("scikit-image", "skimage"),
        ("matplotlib", "matplotlib"),
        ("joblib", "joblib"),
        ("Pillow", "PIL"),
        ("tqdm", "tqdm"),
    ]
    
    results = []
    
    for display_name, import_name in modules:
        results.append(test_import(display_name, import_name))
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All dependencies installed successfully!")
        print("You're ready to use the Plant Leaf Disease Detection System")
    else:
        print("✗ Some dependencies are missing")
        print("Run: pip install -r requirements.txt")
    print("="*60 + "\n")
    
    # Test Python version
    print("Python Version:")
    print(f"  {sys.version}")
    
    if sys.version_info >= (3, 8):
        print("  ✓ Python version is compatible (3.8+)")
    else:
        print("  ✗ Python version should be 3.8 or higher")
    
    print()

if __name__ == "__main__":
    main()