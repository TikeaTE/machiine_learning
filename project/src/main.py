from prepare_data import load_and_merge
from compare import run_comparison

if __name__ == '__main__':
    print('Step 1: Preparing data...\n')
    load_and_merge()

    print('\nStep 2: Running models and comparison...\n')
    run_comparison()

    print('\nDone! All plots saved to project/outputs/plots/')
