from bts_nvs.contest import CURRENT_DATASET_ID, CURRENT_DATASET_ROOT_NAME


def test_current_dataset_constants_identify_round2_input_without_phase1_aliases():
    assert CURRENT_DATASET_ID == "vai_nvs_round2"
    assert CURRENT_DATASET_ROOT_NAME == "VAI_NVS_DATA_ROUND2"
