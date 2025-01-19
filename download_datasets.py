

import kagglehub
kagglehub.login()






target_directory = "./"








lucamodica_meld_test_muse_path = kagglehub.dataset_download(
    'lucamodica/meld-test-muse'
)

print(lucamodica_meld_test_muse_path)

print('Data source import complete.')