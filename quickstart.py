"""
Quick Start Script - Visiting Card Classifier V2
Run this to train and evaluate your V2 model easily
"""

import os
import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

def check_environment():
    """Checks if the script is running in the intended virtual environment."""
    # Heuristic to find the venv folder
    venv_path = Path(__file__).resolve().parents[1] / ".venv-py312"
    if not venv_path.exists():
        return True # Cannot find venv, so cannot check.

    expected_python = venv_path / "Scripts" / "python.exe"
    current_python = Path(sys.executable)

    if not current_python.samefile(expected_python):
        print(f"\n{'='*70}\n" + "⚠ ENVIRONMENT MISMATCH ⚠".center(70) + f"\n{'='*70}")
        print(f"\nThis script is NOT running from the correct virtual environment.")
        print(f"\nCurrent Python: {current_python}")
        print(f"Expected Python: {expected_python}")
        print("\nTo fix this in VS Code:")
        print("1. Press Ctrl+Shift+P")
        print("2. Type 'Python: Select Interpreter'")
        print(f"3. Select the one at: {expected_python}")
        print("\nThen, run the script again from the VS Code terminal.")
        print(f"\nAlternatively, run this exact command in your terminal:\n")
        print(f'"{expected_python}" "{__file__}"\n')
        return False
    return True

def check_dependencies():
    """Check if all required packages are installed"""
    print("\n" + "="*70)
    print("CHECKING DEPENDENCIES")
    print("="*70)
    
    dependencies = [
        'tensorflow',
        'opencv-python',
        'pillow',
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'openpyxl'
    ]
    
    # tell me the w package names to the names they are imported with
    module_map = {
        "opencv-python": "cv2",
        "pillow": "PIL",
        "scikit-learn": "sklearn",
    }
    
    missing = []
    for dep in dependencies:
        try:
            # Use the map to find the correct module name, otherwise default
            module_name = module_map.get(dep, dep.replace("-", "_"))
            __import__(module_name)
            print(f"✓ {dep}")
        except ImportError:
            print(f"✗ {dep} - MISSING")
            missing.append(dep)
    
    if missing:
        print(f"\n⚠ Missing packages. Install with:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies installed!")
    return True

def check_training_data(training_folder):
    """Check if training data exists"""
    print("\n" + "="*70)
    print("CHECKING TRAINING DATA")
    print("="*70)
    
    training_path = Path(training_folder)
    
    if not training_path.exists():
        print(f"✗ Training folder not found: {training_folder}")
        return False
    
    categories = [
        "Digital visiting card",
        "Handwritten",
        "Physical visiting card",
        "RX Pad",
        "Stamp",
    ]
    total_images = 0
    
    for category in categories:
        cat_path = training_path / category
        if not cat_path.exists():
            print(f"⚠ Category folder missing: {category}")
            return False
        
        images = list(cat_path.glob("*.[jJ][pP][gG]")) + \
                 list(cat_path.glob("*.[jJ][pP][eE][gG]")) + \
                 list(cat_path.glob("*.[pP][nN][gG]")) + \
                 list(cat_path.glob("*.[bB][mM][pP]"))
        
        count = len(images)
        total_images += count
        
        if count < 10:
            print(f"⚠ {category}: {count} images (recommended: 50+)")
        else:
            print(f"✓ {category}: {count} images")
    
    if total_images == 0:
        print("\n✗ No training images found!")
        return False
    
    print(f"\n✓ Total training images: {total_images}")
    return True

def main():
    """Main execution"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "   VISITING CARD CLASSIFIER V2 - QUICK START  ".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "="*68 + "╝")
    
    # 0. Check environment
    if not check_environment():
        sys.exit(1)

    # Paths
    script_dir = Path(__file__).parent
    training_folder = script_dir / "Training module_2"
    model_folder = script_dir / "trained_model_2"
    
    # 1. Check dependencies
    if not check_dependencies():
        print("\n✗ Please install missing dependencies and try again")
        return
    
    # 2. Check training data
    if not check_training_data(training_folder):
        print("\n✗ Please organize training data in the Training module_2 folder")
        print(f"  Path: {training_folder}")
        return
    
    # 3. Ask user what to do
    print("\n" + "="*70)
    print("WHAT WOULD YOU LIKE TO DO?")
    print("="*70)
    print("\n1. Train V2 model (using data in Training module_2/)")
    print("2. Test on sample images (using trained_model_2/)")
    print("3. Test on custom folder")
    print("4. Exit")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        print("\n" + "="*70)
        print("STARTING MODEL TRAINING V2")
        print("="*70)
        
        from train_custom_model_v2 import train as train_v2
        
        try:
            test_accuracy = train_v2(
                training_folder=str(training_folder),
                output_dir=str(model_folder)
            )
            
            print("\n" + "="*70)
            print("✓ TRAINING COMPLETED SUCCESSFULLY!")
            print("="*70)
            print(f"\nModel saved to: {model_folder}")
            print(f"Final Test Accuracy: {test_accuracy*100:.2f}%")
            print("\nNext steps:")
            print("1. Check training_history_v2.png for training progress")
            print("2. Run this script again to test the model")
            
        except Exception as e:
            print(f"\n✗ Training failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "2":
        print("\n" + "="*70)
        print("TESTING ON SAMPLE IMAGES")
        print("="*70)
        
        from hugging_face_card_classifier_v2 import classify_folder as classify_folder_v2
        
        if not model_folder.exists():
            print("✗ Trained model directory not found!")
            print(f"  Expected at: {model_folder}")
            print("  Please train the V2 model first by selecting option 1.")
            return
        
        try:
            sample_folder = script_dir / "sample_cards"
            if sample_folder.exists():
                print(f"\nClassifying images in: {sample_folder}\n")
                output_excel = sample_folder / "classifier_results_v2.xlsx"
                
                df = classify_folder_v2(
                    folder_path=str(sample_folder),
                    model_folder=str(model_folder),
                    output_excel_path=str(output_excel)
                )
                
                if not df.empty:
                    print("\nResults Summary:")
                    print(df['final_category'].value_counts())
                    print("\nSample results:")
                    print(df[['image_name', 'final_category', 'final_confidence']].head(10))
                    print(f"\n✓ Results saved to: {output_excel}")
            else:
                print(f"⚠ Sample folder not found: {sample_folder}")
        
        except Exception as e:
            print(f"✗ Testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "3":
        print("\n" + "="*70)
        print("TESTING ON CUSTOM FOLDER")
        print("="*70)
        
        test_folder = input("\nEnter path to test folder: ").strip()
        
        if not Path(test_folder).exists():
            print(f"✗ Folder not found: {test_folder}")
            return
        
        from hugging_face_card_classifier_v2 import classify_folder as classify_folder_v2
        
        if not model_folder.exists():
            print("✗ Trained model directory not found!")
            return
        
        try:
            output_file = Path(test_folder) / "predictions.xlsx"
            df = classify_folder_v2(
                folder_path=test_folder,
                model_folder=str(model_folder),
                output_excel_path=str(output_file)
            )
            
            if not df.empty:
                print("\nResults Summary:")
                print(df['final_category'].value_counts())
                print("\nTop results:")
                print(df[['image_name', 'final_category', 'final_confidence']].head(10))
                print(f"\n✓ Results saved to: {output_file}")
        
        except Exception as e:
            print(f"✗ Testing failed: {e}")
            import traceback
            traceback.print_exc()
    
    elif choice == "4":
        print("\nGoodbye!")
        return
    
    else:
        print("\n✗ Invalid choice")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Interrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
