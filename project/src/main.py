from prepare_data import load_and_merge
from eda import run_eda
from compare import run_comparison

if __name__ == '__main__':
    print('Step 1: Preparing data...\n')
    load_and_merge()

    print('\nStep 2: Running EDA...\n')
    run_eda()

    print('\nStep 3: Running models and comparison...\n')
    run_comparison()

    print('\nDone! All plots saved to project/outputs/plots/')
