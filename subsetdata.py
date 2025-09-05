import json
import random

def create_subset_dataset(input_file: str, output_file: str, calib_size: int = 1000, test_size: int = 1000):
    """Create subset of MCQA dataset with specified calibration and test sizes"""
    
    print(f"Loading dataset from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    print(f"Original dataset: {dataset['total_samples']} total samples")
    print(f"Original calibration: {dataset['calibration_samples']} samples")
    print(f"Original test: {dataset['test_samples']} samples")
    
    # Combine all samples
    all_samples = dataset['calibration'] + dataset['test']
    print(f"Combined samples: {len(all_samples)}")
    
    # Shuffle to randomize
    random.shuffle(all_samples)
    
    # Check if we have enough samples
    required_total = calib_size + test_size
    if len(all_samples) < required_total:
        print(f"Warning: Only {len(all_samples)} samples available, but {required_total} requested")
        calib_size = min(calib_size, len(all_samples))
        test_size = min(test_size, len(all_samples) - calib_size)
    
    # Split into calibration and test
    new_calibration = all_samples[:calib_size]
    new_test = all_samples[calib_size:calib_size + test_size]
    
    # Update IDs to be sequential
    for i, item in enumerate(new_calibration + new_test, 1):
        item['id'] = i
    
    # Create new dataset
    new_dataset = {
        "name": "DialFact-MCQA-Subset",
        "description": f"Subset of DialFact MCQA dataset with {calib_size} calibration and {test_size} test samples",
        "version": "1.0",
        "total_samples": len(new_calibration) + len(new_test),
        "calibration_samples": len(new_calibration),
        "test_samples": len(new_test),
        "calibration": new_calibration,
        "test": new_test
    }
    
    # Save to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(new_dataset, f, indent=2, ensure_ascii=False)
    
    print(f"\n=== Subset Dataset Created ===")
    print(f"New calibration samples: {new_dataset['calibration_samples']}")
    print(f"New test samples: {new_dataset['test_samples']}")
    print(f"Total samples: {new_dataset['total_samples']}")
    print(f"Saved to: {output_file}")
    
    # Print sample statistics
    print(f"\n=== Sample Statistics ===")
    
    # Count answer distribution in calibration
    calib_answers = [item['correct_answer'] for item in new_calibration]
    calib_dist = {ans: calib_answers.count(ans) for ans in ['A', 'B', 'C']}
    print(f"Calibration answer distribution: {calib_dist}")
    
    # Count answer distribution in test
    test_answers = [item['correct_answer'] for item in new_test]
    test_dist = {ans: test_answers.count(ans) for ans in ['A', 'B', 'C']}
    print(f"Test answer distribution: {test_dist}")
    
    # Show sample questions
    print(f"\n=== Sample Questions ===")
    if new_calibration:
        sample = new_calibration[0]
        print(f"Sample Calibration Question:")
        print(f"ID: {sample['id']}")
        print(f"Question: {sample['question'][:200]}...")
        print(f"Correct Answer: {sample['correct_answer']}")
        print(f"Evidence pages: {len(sample['search_results'])}")
    
    return new_dataset

def main():
    # Configuration
    input_file = "dialfact_mcqa.json"
    output_file = "dialfact_mcqa_subset.json"
    calib_size = 1000
    test_size = 1000
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Check if input file exists
    import os
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found!")
        print("Make sure you have run the conversion script first.")
        return
    
    # Create subset
    try:
        subset_dataset = create_subset_dataset(input_file, output_file, calib_size, test_size)
        print(f"\nSubset creation completed successfully!")
        
    except Exception as e:
        print(f"Error creating subset: {str(e)}")

if __name__ == "__main__":
    main()