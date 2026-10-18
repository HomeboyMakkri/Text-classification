"""Selection and manual review contracts, including zero errors and duplicates."""

from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from finnews_sentiment.data.load_data import CLASS_ORDER
from finnews_sentiment.models.error_analysis import build_validation_table
from finnews_sentiment.models.failure_cases import (
    ANNOTATION_COLUMNS, load_failure_annotations, select_validation_examples,
    validate_failure_annotations,
)


@pytest.fixture
def table() -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for i, true_label in enumerate(CLASS_ORDER):
        for j, pred_label in enumerate(CLASS_ORDER):
            if i == j:
                continue
            for n in range(5):
                row_id = len(records) + 10
                records.append({
                    'source_row_id': row_id, 'group_id': row_id,
                    'text': f'Raw {row_id}', 'text_clean': f'raw {row_id}',
                    'true_label_id': i, 'pred_label_id': j,
                    'true_label': true_label, 'pred_label': pred_label, 'is_error': True,
                })
    return pd.DataFrame(records)


def _annotations(examples: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        'source_row_id': examples['source_row_id'].tolist(),
        'primary_category': ['other/unclear'] * len(examples),
        'observation': ['Visible observation'] * len(examples),
        'hypothesis': [''] * len(examples),
        'rationale': ['No cause established'] * len(examples),
    }, columns=list(ANNOTATION_COLUMNS))


def test_selection_is_stable_and_capped(table: pd.DataFrame) -> None:
    before = table.copy(deep=True)
    selected = select_validation_examples(table)
    shuffled = select_validation_examples(table.sample(frac=1, random_state=42))
    pd.testing.assert_frame_equal(selected.examples, shuffled.examples)
    pd.testing.assert_frame_equal(selected.pairs, shuffled.pairs)
    assert len(selected.examples) == 18
    assert selected.examples['source_row_id'].tolist() == [10, 11, 12, 15, 16, 17, 20, 21, 22, 25, 26, 27, 30, 31, 32, 35, 36, 37]
    assert selected.pairs['available_errors'].tolist() == [5] * 6
    assert selected.pairs['selected_examples'].tolist() == [3] * 6
    pd.testing.assert_frame_equal(table, before)


def test_groups_are_counted_without_dropping_rows(table: pd.DataFrame) -> None:
    table.loc[1, ['group_id', 'text', 'text_clean']] = [10, 'Raw 10', 'raw 10']
    correct = table.iloc[:1].copy()
    correct['source_row_id'] = 99
    correct['pred_label_id'] = correct['true_label_id']
    correct['pred_label'] = correct['true_label']
    correct['is_error'] = False
    table = pd.concat([table, correct], ignore_index=True)
    result = select_validation_examples(table)
    duplicated = result.examples.loc[result.examples['group_id'].eq(10)]
    assert len(duplicated) == 2
    assert duplicated['group_validation_rows'].tolist() == [3, 3]
    assert duplicated['group_error_rows'].tolist() == [2, 2]
    assert duplicated['group_selected_rows'].tolist() == [2, 2]
    assert duplicated['is_duplicate_group'].all()
    assert result.examples['group_id'].nunique() == 17
    assert result.pairs.iloc[0]['selected_groups'] == 2
    assert 99 not in result.examples['source_row_id'].tolist()


@pytest.mark.parametrize('empty', [True, False])
def test_no_errors_and_empty_annotations(table: pd.DataFrame, empty: bool) -> None:
    table['pred_label_id'] = table['true_label_id']
    table['pred_label'] = table['true_label']
    table['is_error'] = False
    if empty:
        table = table.iloc[:0]
    selected = select_validation_examples(table)
    assert selected.examples.empty
    assert selected.pairs['status'].tolist() == ['empty'] * 6
    assert selected.pairs['available_errors'].sum() == 0
    assert validate_failure_annotations(selected.examples, _annotations(selected.examples)).empty


def test_single_pair_reports_empty_pairs(table: pd.DataFrame) -> None:
    selected = select_validation_examples(table.iloc[:2])
    assert selected.examples['source_row_id'].tolist() == [10, 11]
    assert selected.pairs['status'].tolist() == ['selected'] + ['empty'] * 5


def test_annotations_join_by_id_and_preserve_blank_hypothesis(table: pd.DataFrame, tmp_path: Path) -> None:
    examples = select_validation_examples(table).examples
    annotations = _annotations(examples).iloc[::-1]
    before = annotations.copy(deep=True)
    path = tmp_path / 'annotations.csv'
    annotations.to_csv(path, index=False)
    reviewed = load_failure_annotations(path, examples)
    assert reviewed['source_row_id'].tolist() == examples['source_row_id'].tolist()
    assert reviewed['hypothesis'].eq('').all()
    assert reviewed['observation'].eq('Visible observation').all()
    pd.testing.assert_frame_equal(annotations, before)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'duplicate', 'unknown_category', 'blank_observation', 'blank_rationale', 'null_hypothesis', 'schema', 'fractional_id'])
def test_rejects_bad_annotations(table: pd.DataFrame, damage: str) -> None:
    examples = select_validation_examples(table).examples
    annotations = _annotations(examples)
    if damage == 'missing':
        annotations = annotations.iloc[1:]
    elif damage == 'extra':
        annotations = pd.concat([annotations, annotations.iloc[:1].assign(source_row_id=999)])
    elif damage == 'duplicate':
        annotations.loc[1, 'source_row_id'] = annotations.loc[0, 'source_row_id']
    elif damage == 'unknown_category':
        annotations.loc[0, 'primary_category'] = 'invented'
    elif damage == 'blank_observation':
        annotations.loc[0, 'observation'] = ' '
    elif damage == 'blank_rationale':
        annotations.loc[0, 'rationale'] = ''
    elif damage == 'null_hypothesis':
        annotations.loc[0, 'hypothesis'] = None
    elif damage == 'schema':
        annotations = annotations.drop(columns='rationale')
    else:
        annotations['source_row_id'] = annotations['source_row_id'].astype(float) + 0.5
    with pytest.raises(ValueError):
        validate_failure_annotations(examples, annotations)


def test_train_test_ids_cannot_enter_review() -> None:
    validation = pd.DataFrame({'source_row_id': [4], 'text': ['Raw'], 'text_clean': ['raw'], 'label_id': [0]})
    manifest = pd.DataFrame({'source_row_id': [1, 4, 9], 'group_id': [1, 4, 9], 'split': ['train', 'validation', 'test']})
    predictions = pd.DataFrame({'source_row_id': [4], 'true_label_id': [0], 'selected_pred_label_id': [1], 'day03_baseline_pred_label_id': [0]})
    table = build_validation_table(validation, manifest, predictions)
    examples = select_validation_examples(table).examples
    assert examples['source_row_id'].tolist() == [4]
    for outside_id in (1, 9):
        annotations = _annotations(examples).assign(source_row_id=outside_id)
        with pytest.raises(ValueError, match='Annotation IDs mismatch'):
            validate_failure_annotations(examples, annotations)


@pytest.mark.parametrize('damage', ['duplicate', 'unknown_label', 'error_flag', 'blank_text', 'name'])
def test_rejects_invalid_analysis(table: pd.DataFrame, damage: str) -> None:
    if damage == 'duplicate':
        table.loc[1, 'source_row_id'] = table.loc[0, 'source_row_id']
    elif damage == 'unknown_label':
        table.loc[0, 'true_label_id'] = 3
    elif damage == 'error_flag':
        table.loc[0, 'is_error'] = False
    elif damage == 'blank_text':
        table.loc[0, 'text_clean'] = ' '
    else:
        table.loc[0, 'true_label'] = 'positive'
    with pytest.raises(ValueError):
        select_validation_examples(table)
