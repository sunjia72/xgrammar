from __future__ import annotations

import copy
import csv
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

SCRIPT = Path(__file__).with_name("run_xgrammar_benchmark.py")
SPEC = importlib.util.spec_from_file_location("xgrammar_benchmark_runner", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot import {SCRIPT}")
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class TinyTokenizer:
    """Five real IDs plus one model-logit slot absent from the tokenizer."""

    eos_token_id = 3
    pad_token_id = 4
    all_special_ids = [3, 4]

    def __len__(self) -> int:
        return 5

    def get_vocab(self) -> dict[str, int]:
        return {
            "other-zero": 0,
            "keyword": 1,
            "other-two": 2,
            "<eos>": 3,
            "<pad>": 4,
        }


def _digest(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _compiled_payload(
    *, provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    # The wildcard accepts either the named class or the residual "other"
    # class for one content token, followed by EOS.
    body: dict[str, Any] = {
        "schema_version": runner.COMPILED_SCHEMA_VERSION,
        "kind": "compiled_token_partition_nfa",
        "token_classes": [
            {"name": "keyword", "symbol_id": 0, "token_ids": [1]},
        ],
        "nfa": {
            "num_states": 3,
            "alphabet_size": 3,
            "initials": [0],
            "finals": [2],
            "transitions": [[1, 2, 2]],
            "wildcard_transitions": [[0, 1]],
        },
        "other_symbol_id": 1,
        "stop_symbol_id": 2,
        "n_low": runner.EXPECTED_N_LOW,
        "n": runner.EXPECTED_N,
        "length_contract": runner.EXPECTED_LENGTH_CONTRACT,
        "prompt_token_ids": [0],
        "tokenizer_fingerprint": {
            "vocab_size": 5,
            "eos_token_id": 3,
            "pad_token_id": 4,
            "all_special_ids": [3, 4],
        },
        "provenance": {} if provenance is None else provenance,
    }
    return {**body, "sha256": _digest(body)}


def _unrolled_state_count() -> int:
    compiled = runner.CompiledConstraint.parse(_compiled_payload())
    unrolled = runner.unroll_nfa(
        compiled.nfa,
        n=compiled.n,
        n_low=compiled.n_low,
    )
    return sum(len(layer) for layer in unrolled.layers)


TINY_UNROLLED_STATES = _unrolled_state_count()


def _job_payload(
    index: int,
    *,
    family: str = "family_0",
    model_reference: dict[str, Any] | None = None,
) -> dict[str, Any]:
    job_id = f"job_{index:03d}"
    provenance = (
        {}
        if model_reference is None
        else {
            "dataset": "tiny",
            "job_id": job_id,
            "design_index": index,
            "seed": index,
            "model_provenance": model_reference,
        }
    )
    return {
        "job_id": job_id,
        "design_index": index,
        "structural_cell_index": index // 3,
        "replicate_index": index % 3,
        "constraint": family,
        "prompt_token_ids": [0],
        "n_low": runner.EXPECTED_N_LOW,
        "n": runner.EXPECTED_N,
        "nfa_states": 3,
        "unrolled_states": TINY_UNROLLED_STATES,
        "seed": index,
        "dataset": "tiny",
        "tracked_keyword_token_ids": [[1]],
        "selected_keywords": [{"surface": "keyword", "token_ids": [1]}],
        "compiled_constraint": _compiled_payload(provenance=provenance),
    }


def _model_provenance() -> dict[str, Any]:
    def identity(name: str) -> dict[str, Any]:
        body = {
            "source": name,
            "local": True,
            "resolved_path": f"/fake/{name}",
            "entries": [],
            "entries_sha256": _digest([]),
        }
        return {**body, "identity_sha256": _digest(body)}

    body = {
        "model_profile": "tiny",
        "base_model": identity("base"),
        "hmm_model": identity("hmm"),
    }
    return {**body, "sha256": _digest(body)}


def _execution_selection(
    rows: list[dict[str, Any]],
    *,
    replicate_indices: list[int] | None = None,
) -> dict[str, Any]:
    selected = list(rows)
    if replicate_indices is not None:
        selected = [
            row
            for row in selected
            if row.get("replicate_index") in replicate_indices
        ]
    ids = [str(row["job_id"]) for row in selected]
    body = {
        "schema_version": 1,
        "selection_order": "canonical_workload_order",
        "operation_order": ["replicate_filter", "max_jobs_prefix"],
        "replicate_indices": replicate_indices,
        "max_jobs": None,
        "source_instances": len(rows),
        "instances_after_replicate_filter": len(selected),
        "selected_instances": len(selected),
        "selected_family_counts": dict(
            Counter(str(row["constraint"]) for row in selected)
        ),
        "selected_job_ids": ids,
        "selected_job_ids_sha256": _digest(ids),
    }
    return {**body, "sha256": _digest(body)}


def _workload_payload() -> dict[str, Any]:
    model_provenance = _model_provenance()
    model_reference = runner._model_provenance_reference(model_provenance)
    rows = [
        _job_payload(
            index,
            family=f"family_{index % 10}",
            model_reference=model_reference,
        )
        for index in range(500)
    ]
    counts = Counter(str(row["constraint"]) for row in rows)
    return {
        "kind": "compiled_keyword_dataset",
        "schema_version": runner.WORKLOAD_SCHEMA_VERSION,
        "dataset": "tiny",
        "benchmark_name": "tiny-current-schema",
        "total_instances": 500,
        "length_interval_including_eos": [
            runner.EXPECTED_N_LOW,
            runner.EXPECTED_N,
        ],
        "length_contract": runner.EXPECTED_LENGTH_CONTRACT,
        "model_provenance": model_provenance,
        "family_counts": dict(counts),
        "jobs": rows,
        "jobs_sha256": _digest(rows),
        "execution_selection": _execution_selection(rows),
    }


def _write_payload(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _parsed_job(index: int = 0) -> Any:
    return runner.Job.parse(_job_payload(index), index)


def _progress_metrics() -> dict[str, Any]:
    return {
        "grammar_build_runtime_s": 0.1,
        "compile_runtime_s": 0.2,
        "xgrammar_precompute_runtime_s": 0.3,
        "setup_runtime_s": 0.3,
        "grammar_rule_count": 5,
        "grammar_edge_count": 3,
        "grammar_bytes": 128,
        "compiled_grammar_memory_bytes": 256,
        "model_vocab_size": 6,
    }


def _progress_payload(job: Any, phase: str) -> dict[str, Any]:
    metrics = _progress_metrics()
    if phase in {"mask_audit_complete", "model_loaded"}:
        metrics["initial_mask_audit"] = {"ok": True}
    if phase == "model_loaded":
        metrics["model_load_runtime_s"] = 0.4
    return {
        "schema_version": runner.RESULT_SCHEMA_VERSION,
        **job.identity(),
        "phase": phase,
        "metrics": metrics,
    }


def _successful_worker_result(job: Any) -> dict[str, Any]:
    return {
        **job.identity(),
        "method": runner.METHOD,
        "status": "success",
        "failure_kind": None,
        "tokenizer_config_load_runtime_s": 0.05,
        "model_load_runtime_s": 0.4,
        **_progress_metrics(),
        "generation_runtime_s": 0.1,
        "worker_total_runtime_s": 1.0,
        "valid_generation": True,
        "prompt_token_count": len(job.compiled.prompt_token_ids),
        "generated_token_ids": [1, 3],
        "generated_total_len": 2,
        "generated_content_token_ids": [1],
        "generated_content_len": 1,
        "canonical_eos_token_id": 3,
        "terminal_token_ids": [3],
        "terminal_eos_token_id": 3,
        "terminal_eos_count": 1,
        "terminated_with_eos": True,
        "hit_content_token_cap": False,
        "keyword_occurrence_counts": [1],
        "generated_text": "keyword",
        "initial_mask_audit": {"ok": True},
    }


def _successful_status(job: Any, contract_sha256: str) -> dict[str, Any]:
    progress = _progress_payload(job, "model_loaded")
    return {
        **job.identity(),
        "run_contract_sha256": contract_sha256,
        "status": "success",
        "failure_kind": None,
        "timeout_s": 30,
        "precompute_timeout_s": 10,
        "generation_timeout_s": 20,
        "timed_out": False,
        "timeout_phase": None,
        "progress_phase": progress["phase"],
        "progress_metrics": progress["metrics"],
        "result": _successful_worker_result(job),
    }


def test_compiled_constraint_parses_and_rejects_authenticated_malformation() -> None:
    payload = _compiled_payload()
    compiled = runner.CompiledConstraint.parse(payload)

    assert compiled.sha256 == payload["sha256"]
    assert compiled.nfa.m == 3
    assert compiled.nfa.initials == [0]
    assert compiled.nfa.finals == [2]
    assert compiled.token_classes == (
        runner.TokenClass("keyword", 0, (1,)),
    )
    assert (compiled.other_symbol_id, compiled.stop_symbol_id) == (1, 2)
    assert (compiled.n_low, compiled.n) == (
        runner.EXPECTED_N_LOW,
        runner.EXPECTED_N,
    )

    tampered = copy.deepcopy(payload)
    tampered["n"] = 3
    with pytest.raises(ValueError, match="digest mismatch"):
        runner.CompiledConstraint.parse(tampered)

    malformed = copy.deepcopy(payload)
    malformed["token_classes"][0]["token_ids"] = [1, 1]
    body = {key: value for key, value in malformed.items() if key != "sha256"}
    malformed["sha256"] = _digest(body)
    with pytest.raises(ValueError, match="ordered, disjoint, and nonempty"):
        runner.CompiledConstraint.parse(malformed)


def test_token_partition_bounded_grammar_acceptance_and_full_initial_mask() -> None:
    compiled = runner.CompiledConstraint.parse(_compiled_payload())
    artifact = runner.token_nfa_from_compiled(
        TinyTokenizer(), model_vocab_size=6, compiled=compiled
    )

    assert artifact.token_symbol_ids == (1, 0, 1, 2, -1, -1)
    assert artifact.symbol_token_ids == ((1,), (0, 2), (3,))
    assert artifact.valid_content_token_ids == (0, 1, 2)
    assert artifact.non_eos_special_token_ids == (4,)

    ebnf, metadata = runner.nfa_to_bounded_ebnf(
        artifact, n_low=compiled.n_low, n=compiled.n
    )
    assert "root ::= q_0_0" in ebnf
    assert 'q_1_1 ::= ""' in ebnf
    assert metadata == runner.GrammarMetadata(
        live_state_count=2,
        rule_count=5,
        edge_count=3,
        initial_symbol_ids=(0, 1),
        initial_allows_eos=False,
    )

    grammar = runner.xgr.Grammar.from_ebnf(ebnf)
    compiled_grammar = runner.xgr.GrammarCompiler(
        artifact.tokenizer_info,
        max_threads=1,
        cache_enabled=False,
    ).compile_grammar(grammar)
    mask_audit = runner.audit_initial_mask(
        compiled_grammar, artifact, metadata
    )
    assert mask_audit == {
        "ok": True,
        "allowed_count": 3,
        "expected_count": 3,
        "missing_expected": [],
        "unexpected_allowed": [],
        "stop_token_ids": [3],
    }

    for token_ids in ([0, 3], [1, 3], [2, 3]):
        assert runner.accepts_generation(artifact, token_ids, 2, 2)
    for token_ids in (
        [4, 3],  # non-EOS special token
        [5, 3],  # model logit slot absent from the tokenizer
        [1],
        [1, 3, 3],
        [3, 1],
    ):
        assert not runner.accepts_generation(artifact, token_ids, 2, 2)


def test_model_native_terminal_aliases_share_the_stop_class() -> None:
    class AliasTokenizer(TinyTokenizer):
        all_special_ids = [3, 4, 5]

        def __len__(self) -> int:
            return 6

        def get_vocab(self) -> dict[str, int]:
            return {
                **super().get_vocab(),
                "<end_of_turn>": 5,
            }

    payload = _compiled_payload()
    payload["tokenizer_fingerprint"] = {
        "vocab_size": 6,
        "eos_token_id": 3,
        "pad_token_id": 4,
        "all_special_ids": [3, 4, 5],
    }
    payload["sha256"] = _digest(
        {key: value for key, value in payload.items() if key != "sha256"}
    )
    compiled = runner.CompiledConstraint.parse(payload)
    artifact = runner.token_nfa_from_compiled(
        AliasTokenizer(),
        model_vocab_size=7,
        compiled=compiled,
        terminal_token_ids=(3, 5),
    )

    assert artifact.terminal_token_ids == (3, 5)
    assert artifact.token_symbol_ids == (1, 0, 1, 2, -1, 2, -1)
    assert artifact.symbol_token_ids == ((1,), (0, 2), (3, 5))
    assert artifact.valid_content_token_ids == (0, 1, 2)
    assert artifact.non_eos_special_token_ids == (4,)
    assert runner.accepts_generation(artifact, [1, 3], 2, 2)
    assert runner.accepts_generation(artifact, [1, 5], 2, 2)
    assert not runner.accepts_generation(artifact, [5, 3], 2, 2)


def test_current_500_job_workload_loads_and_rejects_reauthenticated_tamper(
    tmp_path: Path,
) -> None:
    payload = _workload_payload()
    workload_path = tmp_path / "workload.json"
    _write_payload(workload_path, payload)

    workload = runner.Workload.load(workload_path)
    assert len(workload.jobs) == 500
    assert workload.dataset == "tiny"
    assert workload.benchmark_name == "tiny-current-schema"
    assert Counter(job.source["constraint"] for job in workload.jobs) == {
        f"family_{index}": 50 for index in range(10)
    }

    tampered = copy.deepcopy(payload)
    tampered["jobs"][0]["n"] = 3
    # Reauthenticate the outer row list so validation reaches the independent
    # row-versus-compiled-artifact consistency check.
    tampered["jobs_sha256"] = _digest(tampered["jobs"])
    tampered_path = tmp_path / "tampered.json"
    _write_payload(tampered_path, tampered)
    with pytest.raises(ValueError, match="row and compiled constraint differ"):
        runner.Workload.load(tampered_path)


def test_compiled_artifact_requires_explicit_terminal_eos_contract() -> None:
    payload = _compiled_payload()
    del payload["length_contract"]
    payload["sha256"] = _digest(
        {key: value for key, value in payload.items() if key != "sha256"}
    )

    with pytest.raises(ValueError, match="length contract"):
        runner.CompiledConstraint.parse(payload)


def test_workload_honors_authenticated_execution_selection(
    tmp_path: Path,
) -> None:
    payload = _workload_payload()
    payload["execution_selection"] = _execution_selection(
        payload["jobs"], replicate_indices=[0]
    )
    path = tmp_path / "selected.json"
    _write_payload(path, payload)

    workload = runner.Workload.load(path)

    assert len(workload.jobs) == 167
    assert {job.source["replicate_index"] for job in workload.jobs} == {0}

    del payload["execution_selection"]
    _write_payload(path, payload)
    with pytest.raises(ValueError, match="execution_selection"):
        runner.Workload.load(path)


@pytest.mark.parametrize(
    "phase",
    ("precompute_complete", "mask_audit_complete", "model_loaded"),
)
def test_validate_progress_accepts_each_monotonic_phase(phase: str) -> None:
    job = _parsed_job()
    payload = _progress_payload(job, phase)

    validated = runner._validate_progress(job, payload)

    assert validated == payload
    assert validated["phase"] == phase
    assert validated["metrics"]["xgrammar_precompute_runtime_s"] == 0.3


def test_validate_progress_rejects_malformed_payloads() -> None:
    job = _parsed_job()

    changed_identity = _progress_payload(job, "precompute_complete")
    changed_identity["job_sha256"] = "tampered"
    with pytest.raises(ValueError, match="worker progress changed job_sha256"):
        runner._validate_progress(job, changed_identity)

    mismatched_setup = _progress_payload(job, "precompute_complete")
    mismatched_setup["metrics"]["setup_runtime_s"] = 0.4
    with pytest.raises(ValueError, match="setup runtime differs"):
        runner._validate_progress(job, mismatched_setup)

    missing_audit = _progress_payload(job, "mask_audit_complete")
    missing_audit["metrics"].pop("initial_mask_audit")
    with pytest.raises(ValueError, match="mask audit is invalid"):
        runner._validate_progress(job, missing_audit)

    invalid_model_runtime = _progress_payload(job, "model_loaded")
    invalid_model_runtime["metrics"]["model_load_runtime_s"] = True
    with pytest.raises(
        ValueError, match="model_load_runtime_s must be nonnegative"
    ):
        runner._validate_progress(job, invalid_model_runtime)

    unknown_phase = _progress_payload(job, "precompute_complete")
    unknown_phase["phase"] = "generation_complete"
    with pytest.raises(ValueError, match="phase is invalid"):
        runner._validate_progress(job, unknown_phase)


def test_load_statuses_validates_nested_success_and_rejects_corruption(
    tmp_path: Path,
) -> None:
    job = _parsed_job()
    contract_sha256 = "run-contract"
    status_path = tmp_path / "jobs" / job.job_id / "status.json"
    valid_status = _successful_status(job, contract_sha256)
    _write_payload(status_path, valid_status)

    loaded = runner._load_statuses(
        tmp_path,
        [job],
        run_contract_sha256=contract_sha256,
        expected_timeouts={
            "timeout_s": 30,
            "precompute_timeout_s": 10,
            "generation_timeout_s": 20,
        },
    )
    assert loaded[job.job_id]["result"] == valid_status["result"]

    corrupt_success = copy.deepcopy(valid_status)
    corrupt_success["result"]["generated_token_ids"] = [4, 3]
    _write_payload(status_path, corrupt_success)
    with pytest.raises(ValueError, match="token sequence is invalid"):
        runner._load_statuses(
            tmp_path, [job], run_contract_sha256=contract_sha256
        )

    unknown_failure = {
        **job.identity(),
        "run_contract_sha256": contract_sha256,
        "status": "failed",
        "failure_kind": "unknown_failure",
        "timeout_s": 30,
        "precompute_timeout_s": 10,
        "generation_timeout_s": 20,
        "timed_out": False,
        "timeout_phase": None,
        "progress_phase": None,
        "progress_metrics": None,
        "result": None,
    }
    _write_payload(status_path, unknown_failure)
    with pytest.raises(ValueError, match="saved failure result is invalid"):
        runner._load_statuses(
            tmp_path, [job], run_contract_sha256=contract_sha256
    )


def test_overlapping_occurrence_count() -> None:
    assert runner._overlapping_occurrence_count([1, 1, 1], [1, 1]) == 2
    assert runner._overlapping_occurrence_count([1, 2], [3]) == 0
    assert runner._overlapping_occurrence_count([1, 2], []) == 0


def test_load_statuses_rejects_corrupt_failed_resume_checkpoint_and_timeout(
    tmp_path: Path,
) -> None:
    job = _parsed_job()
    contract_sha256 = "run-contract"
    status_path = tmp_path / "jobs" / job.job_id / "status.json"
    progress = _progress_payload(job, "mask_audit_complete")
    failed_status = {
        **job.identity(),
        "run_contract_sha256": contract_sha256,
        "status": "failed",
        "failure_kind": "post_precompute_timeout",
        "timeout_s": 30,
        "precompute_timeout_s": 10,
        "generation_timeout_s": 20,
        "timed_out": True,
        "timeout_phase": "post_precompute",
        "progress_phase": progress["phase"],
        "progress_metrics": progress["metrics"],
        "result": None,
    }
    _write_payload(status_path, failed_status)

    loaded = runner._load_statuses(
        tmp_path, [job], run_contract_sha256=contract_sha256
    )
    assert loaded[job.job_id]["failure_kind"] == "post_precompute_timeout"

    corrupt_progress = copy.deepcopy(failed_status)
    corrupt_progress["progress_metrics"]["setup_runtime_s"] = 0.4
    _write_payload(status_path, corrupt_progress)
    with pytest.raises(ValueError, match="setup runtime differs"):
        runner._load_statuses(
            tmp_path, [job], run_contract_sha256=contract_sha256
        )

    inconsistent_timeout = copy.deepcopy(failed_status)
    inconsistent_timeout["timed_out"] = False
    _write_payload(status_path, inconsistent_timeout)
    with pytest.raises(ValueError, match="failure timeout is inconsistent"):
        runner._load_statuses(
            tmp_path, [job], run_contract_sha256=contract_sha256
        )


def test_flat_row_preserves_precompute_metrics_after_post_precompute_timeout() -> None:
    job = _parsed_job()
    status = {
        **job.identity(),
        "status": "failed",
        "failure_kind": "post_precompute_timeout",
        "timeout_s": 30,
        "precompute_timeout_s": 10,
        "generation_timeout_s": 20,
        "timed_out": True,
        "timeout_phase": "post_precompute",
        "progress_phase": "mask_audit_complete",
        "worker_subprocess_runtime_s": 30.5,
        "progress_metrics": {
            **_progress_metrics(),
            "initial_mask_audit": {"ok": True},
        },
        "result": None,
    }

    row = runner._flat_row(job, status)

    assert row["status"] == "failed"
    assert row["failure_kind"] == "post_precompute_timeout"
    assert row["timeout_phase"] == "post_precompute"
    assert row["progress_phase"] == "mask_audit_complete"
    assert row["xgrammar_precompute_runtime_s"] == 0.3
    assert row["setup_runtime_s"] == 0.3
    assert row["grammar_build_runtime_s"] == 0.1
    assert row["compile_runtime_s"] == 0.2
    assert row["grammar_rule_count"] == 5
    assert row["initial_mask_audit"] == {"ok": True}


def test_aggregate_preserves_terminal_and_pending_status_semantics(
    tmp_path: Path,
) -> None:
    jobs = tuple(
        runner.Job.parse(_job_payload(index), index)
        for index in range(3)
    )
    statuses = {
        jobs[0].job_id: {
            **jobs[0].identity(),
            "status": "success",
            "failure_kind": None,
            "timeout_s": 10,
            "worker_subprocess_runtime_s": 1.5,
            "result": {
                # Result payloads cannot overwrite authenticated identity or
                # the launcher's terminal status.
                "job_id": "forged",
                "status": "failed",
                "failure_kind": "worker_error",
                "valid_generation": True,
                "xgrammar_precompute_runtime_s": 0.25,
            },
        },
        jobs[1].job_id: {
            **jobs[1].identity(),
            "status": "failed",
            "failure_kind": "post_precompute_timeout",
            "timeout_s": 10,
            "worker_subprocess_runtime_s": 10.1,
            "result": None,
        },
    }

    summary = runner._aggregate(tmp_path, jobs, statuses)
    assert summary["jobs"] == 3
    assert summary["succeeded"] == 1
    assert summary["failed"] == 1
    assert summary["pending"] == 1
    assert summary["failure_counts"] == {"post_precompute_timeout": 1}
    assert summary["valid_generations"] == 1

    with (tmp_path / "results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["status"] for row in rows] == ["success", "failed", "pending"]
    assert rows[0]["job_id"] == jobs[0].job_id
    assert rows[0]["failure_kind"] == ""
    assert rows[0]["valid_generation"] == "True"
    assert rows[1]["failure_kind"] == "post_precompute_timeout"
    assert rows[2]["failure_kind"] == ""

    persisted = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert persisted == summary
