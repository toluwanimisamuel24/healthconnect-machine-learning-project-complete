import pandas as pd
import pytest

from src import data_ingestion


def _write_csv(tmp_path, df: pd.DataFrame, name="fixture.csv"):
    path = tmp_path / name
    df.to_csv(path, index=False)
    return path


def _valid_fixture_df():
    """A minimal DataFrame containing every column data_ingestion requires."""
    return pd.DataFrame({col: [] for col in data_ingestion.EXPECTED_COLUMNS})


def test_load_raw_data_success(tmp_path):
    fixture_df = _valid_fixture_df()
    # add one real row so it isn't empty
    fixture_df.loc[0] = ["A-1", "P-1", "Male", 30, "18-30", "General", "1/1/2025",
                          "1/10/2025", "Monday", "09:00", 9, 0, 0, "Yes", "SMS",
                          5.0, 15.0, "Attended"]
    path = _write_csv(tmp_path, fixture_df)

    result = data_ingestion.load_raw_data(path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    assert set(data_ingestion.EXPECTED_COLUMNS).issubset(result.columns)


def test_load_raw_data_missing_file_raises(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    with pytest.raises(data_ingestion.DataIngestionError, match="not found"):
        data_ingestion.load_raw_data(missing_path)


def test_load_raw_data_empty_file_raises(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    with pytest.raises(data_ingestion.DataIngestionError):
        data_ingestion.load_raw_data(path)


def test_load_raw_data_missing_columns_raises(tmp_path):
    incomplete_df = pd.DataFrame({"appointment_id": ["A-1"], "patient_id": ["P-1"]})
    path = _write_csv(tmp_path, incomplete_df)

    with pytest.raises(data_ingestion.DataIngestionError, match="missing expected columns"):
        data_ingestion.load_raw_data(path)


def test_load_raw_data_does_not_mutate_source_file(tmp_path):
    """Loading should never write back to the source file."""
    fixture_df = _valid_fixture_df()
    fixture_df.loc[0] = ["A-1", "P-1", "Male", 30, "18-30", "General", "1/1/2025",
                          "1/10/2025", "Monday", "09:00", 9, 0, 0, "Yes", "SMS",
                          5.0, 15.0, "Attended"]
    path = _write_csv(tmp_path, fixture_df)
    original_bytes = path.read_bytes()

    data_ingestion.load_raw_data(path)

    assert path.read_bytes() == original_bytes
