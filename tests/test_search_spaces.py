# SPDX-License-Identifier: LGPL-3.0-or-later
"""Tests for MELITE's internal classifier search-space contract."""

import ast
import math
from pathlib import Path

import pytest
import melite.main as main_module
import melite.search_spaces as search_spaces_module
from melite.config import Config
from melite.model_training import MultiModelTrainer
from melite.search_spaces import (
    SEARCH_SPACE_POLICY,
    BranchSelector,
    CategoricalDomain,
    ClassifierSearchSpace,
    FixedParameterSpec,
    FloatDomain,
    IntDomain,
    ParameterSpec,
    SearchBranch,
    get_search_space,
    search_space_from_json,
    search_space_to_json,
)


EXPECTED_BUILTIN_JSON = {
    "svc": (
        '{"branches":[{"fixed_parameters":[],"parameters":[],"selector_value"'
        ':"linear"},{"fixed_parameters":[],"parameters":[{"domain":{"high":'
        '0.2,"log":true,"low":0.001,"type":"float"},"name":"gamma","target"'
        ':"svc__gamma"}],"selector_value":"rbf"},{"fixed_parameters":[],"para'
        'meters":[{"domain":{"high":0.2,"log":true,"low":0.001,"type":"float"'
        '},"name":"gamma","target":"svc__gamma"},{"domain":{"high":1.0,"log"'
        ':false,"low":0.0,"type":"float"},"name":"coef0","target":"svc__coef0"'
        '},{"domain":{"high":5,"low":3,"step":1,"type":"int"},"name":"degree"'
        ',"target":"svc__degree"}],"selector_value":"poly"}],"classifier":"svc"'
        ',"common_parameters":[{"domain":{"high":20.0,"log":true,"low":0.01,'
        '"type":"float"},"name":"C","target":"svc__C"}],"selector":{"choices"'
        ':["linear","rbf","poly"],"name":"kernel","target":"svc__kernel"}}'
    ),
    "rf": (
        '{"branches":[{"fixed_parameters":[{"name":"max_depth","target":"max_'
        'depth","value":null}],"parameters":[],"selector_value":"unbounded"},{"f'
        'ixed_parameters":[],"parameters":[{"domain":{"high":40,"low":10,"step"'
        ':1,"type":"int"},"name":"max_depth","target":"max_depth"}],"selector_v'
        'alue":"bounded"}],"classifier":"rf","common_parameters":[{"domain":{"'
        'choices":[200,400,800],"type":"categorical"},"name":"n_estimators","tar'
        'get":"n_estimators"},{"domain":{"choices":["sqrt","log2"],"type":"cate'
        'gorical"},"name":"max_features","target":"max_features"},{"domain":{"h'
        'igh":5,"low":2,"step":1,"type":"int"},"name":"min_samples_split","tar'
        'get":"min_samples_split"},{"domain":{"high":2,"low":1,"step":1,"type"'
        ':"int"},"name":"min_samples_leaf","target":"min_samples_leaf"}],"sele'
        'ctor":{"choices":["unbounded","bounded"],"name":"depth_mode","target":n'
        "ull}}"
    ),
    "xgb": (
        '{"branches":[{"fixed_parameters":[{"name":"gamma","target":"gamma","'
        'value":0.0}],"parameters":[],"selector_value":"zero"},{"fixed_parameters"'
        ':[],"parameters":[{"domain":{"high":5.0,"log":true,"low":0.01,"type":'
        '"float"},"name":"gamma","target":"gamma"}],"selector_value":"positive"'
        '}],"classifier":"xgb","common_parameters":[{"domain":{"high":600,"low"'
        ':300,"step":1,"type":"int"},"name":"n_estimators","target":"n_estimato'
        'rs"},{"domain":{"high":0.1,"log":true,"low":0.01,"type":"float"},"nam'
        'e":"learning_rate","target":"learning_rate"},{"domain":{"high":8,"low"'
        ':4,"step":1,"type":"int"},"name":"max_depth","target":"max_depth"},{"'
        'domain":{"high":0.85,"log":false,"low":0.7,"type":"float"},"name":"su'
        'bsample","target":"subsample"},{"domain":{"high":1.0,"log":false,"low"'
        ':0.7,"type":"float"},"name":"colsample_bytree","target":"colsample_byt'
        'ree"},{"domain":{"high":0.5,"log":false,"low":0.0,"type":"float"},"n'
        'ame":"reg_alpha","target":"reg_alpha"},{"domain":{"high":2.0,"log":f'
        'alse,"low":1.0,"type":"float"},"name":"reg_lambda","target":"reg_lamb'
        'da"}],"selector":{"choices":["zero","positive"],"name":"gamma_mode","t'
        'arget":null}}'
    ),
}


def _simple_parameter(name="alpha", target="alpha"):
    return ParameterSpec(name, target, FloatDomain(0.1, 1.0))


def _two_branch_space(
    *,
    common_parameters=(),
    selector=None,
    first_fixed=(),
    first_parameters=(),
    second_fixed=(),
    second_parameters=(),
):
    selector = selector or BranchSelector("choice", "choice", ("a", "b"))
    return ClassifierSearchSpace(
        classifier="test",
        common_parameters=common_parameters,
        selector=selector,
        branches=(
            SearchBranch("a", first_fixed, first_parameters),
            SearchBranch("b", second_fixed, second_parameters),
        ),
    )


def test_stack_is_known_and_deliberately_non_tunable():
    assert get_search_space("stack") is None


def test_unknown_classifier_key_raises_key_error():
    with pytest.raises(KeyError, match="Unknown MELITE classifier key"):
        get_search_space("unknown")


def test_float_domain_accepts_arbitrary_positive_log_bounds():
    assert FloatDomain(0.03, 0.7, log=True) == FloatDomain(0.03, 0.7, log=True)


@pytest.mark.parametrize(
    ("args", "error_type", "message"),
    [
        ((True, 1.0), TypeError, "excluding bool"),
        ((0.1, False), TypeError, "excluding bool"),
        ((math.nan, 1.0), ValueError, "finite"),
        ((0.1, math.inf), ValueError, "finite"),
        ((1.0, 1.0), ValueError, "less than"),
        ((2.0, 1.0), ValueError, "less than"),
        ((0.0, 1.0, True), ValueError, "low > 0"),
        ((0.1, 1.0, "yes"), TypeError, "must be bool"),
    ],
)
def test_float_domain_rejects_invalid_contracts(args, error_type, message):
    with pytest.raises(error_type, match=message):
        FloatDomain(*args)


@pytest.mark.parametrize(
    ("args", "error_type", "message"),
    [
        ((True, 5), TypeError, "excluding bool"),
        ((1, 5.0), TypeError, "excluding bool"),
        ((1, 1), ValueError, "less than"),
        ((5, 1), ValueError, "less than"),
        ((1, 5, 0), ValueError, "positive"),
        ((1, 5, -1), ValueError, "positive"),
        ((1, 6, 2), ValueError, "compatible"),
    ],
)
def test_int_domain_rejects_invalid_contracts(args, error_type, message):
    with pytest.raises(error_type, match=message):
        IntDomain(*args)


def test_categorical_domain_preserves_order_and_normalizes_to_tuple():
    domain = CategoricalDomain(["second", "first"])

    assert domain.choices == ("second", "first")


@pytest.mark.parametrize("choices", [{"a", "b"}, frozenset({"a", "b"})])
def test_categorical_domain_explicitly_rejects_unordered_sets(choices):
    with pytest.raises(TypeError, match="set and frozenset"):
        CategoricalDomain(choices)


@pytest.mark.parametrize(
    ("choices", "error_type", "message"),
    [
        (("only",), ValueError, "at least two"),
        (("same", "same"), ValueError, "duplicate"),
        ("ab", TypeError, "ordered sequence"),
        (("ok", object()), TypeError, "JSON-compatible"),
        ((0.0, math.inf), ValueError, "finite"),
    ],
)
def test_categorical_domain_rejects_invalid_choices(choices, error_type, message):
    with pytest.raises(error_type, match=message):
        CategoricalDomain(choices)


def test_parameter_and_fixed_parameter_specs_validate_immediately():
    with pytest.raises(ValueError, match="non-empty"):
        ParameterSpec("", "target", FloatDomain(0.1, 1.0))
    with pytest.raises(ValueError, match="non-empty"):
        ParameterSpec("name", " ", FloatDomain(0.1, 1.0))
    with pytest.raises(TypeError, match="supported domain"):
        ParameterSpec("name", "target", object())
    with pytest.raises(ValueError, match="non-empty"):
        FixedParameterSpec("", "target", 1)
    with pytest.raises(TypeError, match="JSON-compatible"):
        FixedParameterSpec("name", "target", object())


@pytest.mark.parametrize(
    "factory",
    [
        lambda: ParameterSpec(" name", "target", FloatDomain(0.1, 1.0)),
        lambda: ParameterSpec("name", "target ", FloatDomain(0.1, 1.0)),
        lambda: FixedParameterSpec(" name", "target", 1),
        lambda: FixedParameterSpec("name", "target ", 1),
        lambda: BranchSelector(" choice", "choice", ("a", "b")),
        lambda: BranchSelector("choice", "choice ", ("a", "b")),
        lambda: ClassifierSearchSpace(
            " test",
            common_parameters=(_simple_parameter(),),
        ),
    ],
)
def test_contract_names_and_targets_reject_surrounding_whitespace(factory):
    with pytest.raises(ValueError, match="non-empty string"):
        factory()


def test_pure_mode_selector_requires_mode_suffix():
    selector = BranchSelector("depth_mode", None, ("bounded", "unbounded"))

    assert selector.target is None
    with pytest.raises(ValueError, match="ending in '_mode'"):
        BranchSelector("depth", None, ("bounded", "unbounded"))


def test_selector_none_requires_no_branches():
    with pytest.raises(ValueError, match="requires no branches"):
        ClassifierSearchSpace(
            "test",
            selector=None,
            branches=(SearchBranch("unused"),),
        )


def test_tunable_space_without_selector_requires_a_common_parameter():
    with pytest.raises(ValueError, match="requires at least one common parameter"):
        ClassifierSearchSpace("test")


def test_estimator_target_selector_may_have_all_branches_empty():
    space = _two_branch_space()

    assert all(not branch.parameters for branch in space.branches)
    assert all(not branch.fixed_parameters for branch in space.branches)


def test_melite_only_selector_rejects_all_branches_empty():
    with pytest.raises(ValueError, match="MELITE-only selector requires"):
        _two_branch_space(
            selector=BranchSelector("choice_mode", None, ("a", "b")),
        )


def test_melite_only_selector_accepts_one_branch_effect():
    space = _two_branch_space(
        selector=BranchSelector("choice_mode", None, ("a", "b")),
        second_fixed=(FixedParameterSpec("alpha", "alpha", 1.0),),
    )

    assert not space.branches[0].fixed_parameters
    assert space.branches[1].fixed_parameters


def test_selector_requires_branches():
    with pytest.raises(ValueError, match="requires branches"):
        ClassifierSearchSpace(
            "test",
            selector=BranchSelector("choice", "choice", ("a", "b")),
        )


def test_missing_selector_branch_is_rejected():
    with pytest.raises(ValueError, match="Missing branch"):
        ClassifierSearchSpace(
            "test",
            selector=BranchSelector("choice", "choice", ("a", "b")),
            branches=(SearchBranch("a"),),
        )


def test_unknown_branch_selector_value_is_rejected():
    with pytest.raises(ValueError, match="Unknown branch selector_value"):
        ClassifierSearchSpace(
            "test",
            selector=BranchSelector("choice", "choice", ("a", "b")),
            branches=(SearchBranch("a"), SearchBranch("unknown")),
        )


def test_duplicate_selector_branch_is_rejected():
    with pytest.raises(ValueError, match="Duplicate branch"):
        ClassifierSearchSpace(
            "test",
            selector=BranchSelector("choice", "choice", ("a", "b")),
            branches=(SearchBranch("a"), SearchBranch("a")),
        )


def test_logical_name_collision_on_effective_path_is_rejected():
    with pytest.raises(ValueError, match="Logical parameter name collision"):
        _two_branch_space(
            common_parameters=(_simple_parameter("alpha", "common_alpha"),),
            first_parameters=(_simple_parameter("alpha", "branch_alpha"),),
        )


def test_estimator_target_collision_on_effective_path_is_rejected():
    with pytest.raises(ValueError, match="Estimator target collision"):
        _two_branch_space(
            common_parameters=(_simple_parameter("common", "same_target"),),
            first_parameters=(_simple_parameter("branch", "same_target"),),
        )


def test_fixed_and_searchable_logical_name_collision_is_rejected():
    with pytest.raises(ValueError, match="Logical parameter name collision"):
        _two_branch_space(
            first_fixed=(FixedParameterSpec("shared", "fixed_target", 0),),
            first_parameters=(_simple_parameter("shared", "search_target"),),
        )


def test_fixed_and_searchable_estimator_target_collision_is_rejected():
    with pytest.raises(ValueError, match="Estimator target collision"):
        _two_branch_space(
            first_fixed=(FixedParameterSpec("fixed", "shared_target", 0),),
            first_parameters=(_simple_parameter("search", "shared_target"),),
        )


def test_selector_logical_name_collision_is_rejected():
    with pytest.raises(ValueError, match="Logical parameter name collision"):
        _two_branch_space(
            common_parameters=(_simple_parameter("choice", "other_target"),),
        )


def test_selector_estimator_target_collision_is_rejected():
    with pytest.raises(ValueError, match="Estimator target collision"):
        _two_branch_space(
            common_parameters=(_simple_parameter("other_name", "choice"),),
        )


def test_logical_parameter_target_must_match_across_branches():
    with pytest.raises(ValueError, match="inconsistent estimator targets"):
        _two_branch_space(
            first_parameters=(_simple_parameter("alpha", "first_alpha"),),
            second_parameters=(_simple_parameter("alpha", "second_alpha"),),
        )


def test_searchable_parameter_domain_must_match_across_branches():
    with pytest.raises(ValueError, match="inconsistent domains"):
        _two_branch_space(
            first_parameters=(ParameterSpec("alpha", "alpha", FloatDomain(0.1, 1.0)),),
            second_parameters=(ParameterSpec("alpha", "alpha", FloatDomain(0.2, 1.0)),),
        )


def test_parameter_may_be_fixed_in_one_branch_and_searchable_in_another():
    space = _two_branch_space(
        first_fixed=(FixedParameterSpec("alpha", "alpha", 0.0),),
        second_parameters=(_simple_parameter("alpha", "alpha"),),
    )

    assert space.branches[0].fixed_parameters[0].target == "alpha"
    assert space.branches[1].parameters[0].target == "alpha"


def test_fixed_parameter_values_may_differ_across_branches():
    space = _two_branch_space(
        first_fixed=(FixedParameterSpec("alpha", "alpha", 0.0),),
        second_fixed=(FixedParameterSpec("alpha", "alpha", 1.0),),
    )

    assert space.branches[0].fixed_parameters[0].value == 0.0
    assert space.branches[1].fixed_parameters[0].value == 1.0


def test_branch_for_returns_the_existing_branch_object():
    space = _two_branch_space()

    assert space.branch_for("b") is space.branches[1]


def test_branch_for_rejects_an_unknown_selector_value():
    space = _two_branch_space()

    with pytest.raises(KeyError, match="Unknown selector value 'unknown'"):
        space.branch_for("unknown")


def test_branch_for_rejects_a_space_without_a_selector():
    space = ClassifierSearchSpace(
        "test",
        common_parameters=(_simple_parameter(),),
    )

    with pytest.raises(ValueError, match="does not define a branch selector"):
        space.branch_for("a")


@pytest.mark.parametrize("classifier_key", ["svc", "rf", "xgb"])
def test_search_space_canonical_json_round_trip_is_stable(classifier_key):
    original = get_search_space(classifier_key)
    assert original is not None

    first_json = search_space_to_json(original)
    restored = search_space_from_json(first_json)
    second_json = search_space_to_json(restored)

    assert restored == original
    assert second_json == first_json


def test_builtin_search_space_contracts_are_exactly_preserved():
    """Protect MELITE's built-in methodological search-space contract.

    A failure requires reverting accidental drift or deliberately reviewing the
    methodological change and updating this expected representation together
    with the corresponding release documentation.
    """
    actual = {
        classifier_key: search_space_to_json(get_search_space(classifier_key))
        for classifier_key in ("svc", "rf", "xgb")
    }

    assert actual == EXPECTED_BUILTIN_JSON


def test_builtin_classifier_rosters_are_exactly_consistent():
    trainer = MultiModelTrainer(Config())
    expected = {"svc", "rf", "xgb", "stack"}

    assert (
        set(trainer.model_builders)
        == set(main_module._CLASSIFIER_NAMES)
        == set(SEARCH_SPACE_POLICY)
        == expected
    )


def test_search_space_module_has_no_optuna_or_runtime_training_dependency():
    source_path = Path(search_spaces_module.__file__)
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_modules = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                module = "melite"
                if node.module is not None:
                    module = f"{module}.{node.module}"
            elif node.module is not None:
                module = node.module
            else:
                continue
            imported_modules.add(module)
            imported_modules.update(f"{module}.{alias.name}" for alias in node.names)

    forbidden_modules = ("optuna", "melite.model_training", "melite.main")
    for imported_module in imported_modules:
        assert all(
            imported_module != forbidden
            and not imported_module.startswith(f"{forbidden}.")
            for forbidden in forbidden_modules
        )


def test_search_spaces_are_the_only_production_search_contract():
    config = Config()
    trainer = MultiModelTrainer(config)

    assert not hasattr(config, "_param_grid")
    assert not hasattr(config, "_build_param_grid")
    assert not hasattr(trainer, "perform_grid_search")
