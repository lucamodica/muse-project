# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.
import kagglehub
kagglehub.login()

# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.
target_directory = "./"

# Download datasets to the specified directory
# lucamodica_meld_train_muse_path = kagglehub.dataset_download(
#     'lucamodica/meld-train-muse'
# )
# lucamodica_meld_dev_muse_path = kagglehub.dataset_download( 
#     'lucamodica/meld-dev-muse'
# )
lucamodica_meld_test_muse_path = kagglehub.dataset_download(
    'lucamodica/meld-test-muse'
)

print(lucamodica_meld_test_muse_path)

print('Data source import complete.')