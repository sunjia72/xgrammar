#!/usr/bin/env python3
"""Run XGrammar on one authenticated compiled keyword workload.

The runner is deliberately family-agnostic.  It consumes only the method-neutral
``compiled_constraint`` embedded in each schema-v4 workload row: an NFA, a token
partition, EOS-inclusive length bounds, and the exact prompt token IDs.
"""

from __future__ import annotations

import argparse
import atexit
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import queue
import random
import shlex
import signal
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Collection, Mapping, Sequence

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import xgrammar as xgr
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
)


SCRIPT = Path(__file__).resolve()
EXPERIMENT_ROOT = SCRIPT.parent
XGRAMMAR_ROOT = EXPERIMENT_ROOT.parent
NFA_ROOT = XGRAMMAR_ROOT.parent / "nfa_fpras"
NFA_SRC = NFA_ROOT / "src"
if str(NFA_SRC) not in sys.path:
    sys.path.insert(0, str(NFA_SRC))

from nfa_fpras.automata import NFA, WILDCARD, make_nfa, unroll_nfa  # noqa: E402


METHOD = "xgrammar"
RUNNER_VERSION = 5
RESULT_SCHEMA_VERSION = 2
WORKLOAD_KIND = "compiled_keyword_dataset"
WORKLOAD_SCHEMA_VERSION = 4
COMPILED_KIND = "compiled_token_partition_nfa"
COMPILED_SCHEMA_VERSION = 2
TERMINAL_EOS_LENGTH_CONTRACT_SCHEMA_VERSION = 1
EXPECTED_N_LOW = 2
EXPECTED_N = 65
DEFAULT_MODEL = Path("/project/aip-ksmeel/sunjia72/models/Qwen3.5-2B")
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "results_xgrammar_all10_content64_total65"
PRIVATE_CODEPOINT_BASE = 0x10000
MAX_CODEPOINT = 0x10FFFF
_ACTIVE: set[int] = set()
_ACTIVE_LOCK = threading.Lock()
_PROGRESS_PHASES = {
    "precompute_complete": 1,
    "mask_audit_complete": 2,
    "model_loaded": 3,
}
_FAILURE_KINDS = {
    "precompute_timeout",
    "post_precompute_timeout",
    "cuda_oom",
    "invalid_progress",
    "invalid_result",
    "worker_error",
    "launcher_error",
}


def _terminal_eos_length_contract(n_low: int, n: int) -> dict[str, Any]:
    """Mirror the method-neutral compiled-artifact length contract."""

    if not 1 <= n_low <= n:
        raise ValueError("terminal-EOS bounds must satisfy 1 <= n_low <= n")
    return {
        "schema_version": TERMINAL_EOS_LENGTH_CONTRACT_SCHEMA_VERSION,
        "content_token_interval": [n_low - 1, n - 1],
        "total_token_interval_including_eos": [n_low, n],
        "terminal_eos_tokens": 1,
        "eos_counts_toward_total": True,
        "eos_counts_toward_content": False,
    }


EXPECTED_LENGTH_CONTRACT = _terminal_eos_length_contract(
    EXPECTED_N_LOW, EXPECTED_N
)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _selected_job_ids(
    payload: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> tuple[str, ...]:
    """Authenticate and resolve the canonical schema-v4 execution view."""

    selection = payload.get("execution_selection")
    if not isinstance(selection, Mapping):
        raise ValueError("compiled dataset has no valid execution_selection")
    body = dict(selection)
    supplied = body.pop("sha256", None)
    if not isinstance(supplied, str) or supplied != _digest(body):
        raise ValueError("workload execution_selection digest mismatch")
    replicas = body.get("replicate_indices")
    if replicas is not None and (
        not isinstance(replicas, list)
        or any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value < 0
            for value in replicas
        )
        or replicas != sorted(set(replicas))
    ):
        raise ValueError("execution_selection replicate_indices is invalid")
    max_jobs = body.get("max_jobs")
    if max_jobs is not None and (
        isinstance(max_jobs, bool)
        or not isinstance(max_jobs, int)
        or max_jobs <= 0
    ):
        raise ValueError("execution_selection max_jobs is invalid")
    selected = list(rows)
    if replicas is not None:
        selected = [
            row for row in selected if row.get("replicate_index") in replicas
        ]
    after_replicas = len(selected)
    if max_jobs is not None:
        selected = selected[:max_jobs]
    selected_ids = [str(row["job_id"]) for row in selected]
    expected = {
        "schema_version": 1,
        "selection_order": "canonical_workload_order",
        "operation_order": ["replicate_filter", "max_jobs_prefix"],
        "replicate_indices": replicas,
        "max_jobs": max_jobs,
        "source_instances": len(rows),
        "instances_after_replicate_filter": after_replicas,
        "selected_instances": len(selected),
        "selected_family_counts": dict(
            Counter(str(row["constraint"]) for row in selected)
        ),
        "selected_job_ids": selected_ids,
        "selected_job_ids_sha256": _digest(selected_ids),
    }
    if body != expected or not selected_ids:
        raise ValueError("workload execution_selection differs")
    return tuple(selected_ids)


def _file_digest(path: Path) -> str:
    out = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            out.update(block)
    return out.hexdigest()


def _validate_model_snapshot(
    requested_path: Path,
    provenance: Mapping[str, Any],
) -> dict[str, str]:
    """Authenticate the exact base-model snapshot recorded by the workload."""

    provenance_body = dict(provenance)
    provenance_sha256 = provenance_body.pop("sha256", None)
    if provenance_sha256 != _digest(provenance_body):
        raise ValueError("workload model provenance digest mismatch")
    base_model = provenance_body.get("base_model")
    if not isinstance(base_model, Mapping):
        raise ValueError("workload model provenance lacks base_model")
    identity_body = dict(base_model)
    identity_sha256 = identity_body.pop("identity_sha256", None)
    if identity_sha256 != _digest(identity_body):
        raise ValueError("base-model identity digest mismatch")
    resolved = requested_path.expanduser().resolve()
    if (
        identity_body.get("local") is not True
        or identity_body.get("resolved_path") != str(resolved)
        or not resolved.is_dir()
    ):
        raise ValueError("requested base model differs from workload provenance")
    entries = identity_body.get("entries")
    if (
        not isinstance(entries, list)
        or identity_body.get("entries_sha256") != _digest(entries)
    ):
        raise ValueError("base-model entry manifest is invalid")
    for entry in entries:
        if (
            not isinstance(entry, Mapping)
            or not isinstance(entry.get("path"), str)
            or not isinstance(entry.get("bytes"), int)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise ValueError("base-model entry is invalid")
        path = resolved / str(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != int(entry["bytes"])
            or _file_digest(path) != entry["sha256"]
        ):
            raise ValueError(f"base-model snapshot changed: {entry['path']}")
    return {
        "resolved_path": str(resolved),
        "identity_sha256": str(identity_sha256),
        "provenance_sha256": str(provenance_sha256),
    }


def _validate_tokenizer_files(
    model_path: Path,
    fingerprint: Mapping[str, Any],
) -> str:
    files = fingerprint.get("files")
    if (
        not isinstance(files, Mapping)
        or not files
        or fingerprint.get("combined_sha256") != _digest(files)
    ):
        raise ValueError("compiled tokenizer file fingerprint is invalid")
    for filename, expected in files.items():
        if not isinstance(filename, str) or not isinstance(expected, str):
            raise ValueError("compiled tokenizer file entry is invalid")
        path = model_path / filename
        if not path.is_file() or _file_digest(path) != expected:
            raise ValueError(f"runtime tokenizer file differs: {filename}")
    return str(fingerprint["combined_sha256"])


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        return int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an integer") from error


def _integer_tuple(value: Any, name: str, *, nonempty: bool = False) -> tuple[int, ...]:
    if not isinstance(value, list) or (nonempty and not value):
        raise ValueError(f"{name} must be a{' nonempty' if nonempty else ''} list")
    result = tuple(_integer(item, f"{name} item") for item in value)
    return result


def _overlapping_occurrence_count(
    sequence: Sequence[int],
    pattern: Sequence[int],
) -> int:
    if not pattern or len(pattern) > len(sequence):
        return 0
    width = len(pattern)
    target = list(pattern)
    return sum(
        list(sequence[start : start + width]) == target
        for start in range(len(sequence) - width + 1)
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(field for row in rows for field in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            {field: _csv_cell(row.get(field)) for field in fields} for row in rows
        )
    temporary.replace(path)


@dataclass(frozen=True)
class TokenClass:
    name: str
    symbol_id: int
    token_ids: tuple[int, ...]


@dataclass(frozen=True)
class CompiledConstraint:
    nfa: NFA
    token_classes: tuple[TokenClass, ...]
    other_symbol_id: int
    stop_symbol_id: int
    n_low: int
    n: int
    prompt_token_ids: tuple[int, ...]
    tokenizer_fingerprint: Mapping[str, Any]
    provenance: Mapping[str, Any]
    sha256: str

    @classmethod
    def parse(cls, value: Any) -> "CompiledConstraint":
        if not isinstance(value, Mapping):
            raise ValueError("compiled_constraint must be an object")
        body = dict(value)
        supplied = body.pop("sha256", None)
        if body.get("schema_version") != COMPILED_SCHEMA_VERSION:
            raise ValueError("compiled constraint schema mismatch")
        if body.get("kind") != COMPILED_KIND:
            raise ValueError("compiled constraint kind mismatch")
        if not isinstance(supplied, str) or supplied != _digest(body):
            raise ValueError("compiled constraint digest mismatch")
        raw_classes = body.get("token_classes")
        if not isinstance(raw_classes, list) or not raw_classes:
            raise ValueError("compiled token classes are missing")
        classes: list[TokenClass] = []
        seen_tokens: set[int] = set()
        for expected_symbol, raw in enumerate(raw_classes):
            if not isinstance(raw, Mapping):
                raise ValueError("token class must be an object")
            name = raw.get("name")
            symbol = _integer(raw.get("symbol_id"), "token class symbol")
            token_ids = _integer_tuple(raw.get("token_ids"), "token IDs", nonempty=True)
            if (
                not isinstance(name, str)
                or not name
                or symbol != expected_symbol
                or len(set(token_ids)) != len(token_ids)
                or any(token < 0 for token in token_ids)
                or seen_tokens.intersection(token_ids)
            ):
                raise ValueError("token classes must be ordered, disjoint, and nonempty")
            seen_tokens.update(token_ids)
            classes.append(TokenClass(name, symbol, token_ids))
        raw_nfa = body.get("nfa")
        if not isinstance(raw_nfa, Mapping):
            raise ValueError("compiled NFA is missing")
        state_count = _integer(raw_nfa.get("num_states"), "NFA state count")
        alphabet_size = _integer(raw_nfa.get("alphabet_size"), "NFA alphabet size")
        initials = _integer_tuple(raw_nfa.get("initials"), "NFA initials", nonempty=True)
        finals = _integer_tuple(raw_nfa.get("finals"), "NFA finals", nonempty=True)
        explicit = raw_nfa.get("transitions")
        wildcard = raw_nfa.get("wildcard_transitions")
        if not isinstance(explicit, list) or not isinstance(wildcard, list):
            raise ValueError("compiled NFA transition lists are missing")
        edges: list[tuple[int, int, int]] = []
        for edge in explicit:
            if not isinstance(edge, list) or len(edge) != 3:
                raise ValueError("compiled NFA explicit edge is invalid")
            edges.append(tuple(_integer(item, "NFA edge") for item in edge))
        for edge in wildcard:
            if not isinstance(edge, list) or len(edge) != 2:
                raise ValueError("compiled NFA wildcard edge is invalid")
            source, destination = (_integer(item, "NFA wildcard edge") for item in edge)
            edges.append((source, WILDCARD, destination))
        other = _integer(body.get("other_symbol_id"), "other_symbol_id")
        stop = _integer(body.get("stop_symbol_id"), "stop_symbol_id")
        n_low = _integer(body.get("n_low"), "n_low")
        n = _integer(body.get("n"), "n")
        if (
            state_count <= 0
            or alphabet_size != len(classes) + 2
            or other != len(classes)
            or stop != alphabet_size - 1
            or not 1 <= n_low <= n
            or body.get("length_contract")
            != _terminal_eos_length_contract(n_low, n)
        ):
            raise ValueError("compiled NFA alphabet or length contract is invalid")
        nfa = make_nfa(state_count, initials, finals, stop, edges)
        prompt = _integer_tuple(
            body.get("prompt_token_ids"), "prompt_token_ids", nonempty=True
        )
        fingerprint = body.get("tokenizer_fingerprint")
        provenance = body.get("provenance")
        if not isinstance(fingerprint, Mapping) or not isinstance(
            provenance, Mapping
        ):
            raise ValueError(
                "tokenizer fingerprint or artifact provenance is missing"
            )
        return cls(
            nfa,
            tuple(classes),
            other,
            stop,
            n_low,
            n,
            prompt,
            dict(fingerprint),
            dict(provenance),
            supplied,
        )


@dataclass(frozen=True)
class Job:
    source: Mapping[str, Any]
    compiled: CompiledConstraint

    @classmethod
    def parse(cls, value: Any, expected_index: int) -> "Job":
        if not isinstance(value, Mapping):
            raise ValueError("workload job must be an object")
        job_id = value.get("job_id")
        if (
            not isinstance(job_id, str)
            or not job_id
            or "/" in job_id
            or "\\" in job_id
            or value.get("design_index") != expected_index
        ):
            raise ValueError("job ID or design index is invalid")
        compiled = CompiledConstraint.parse(value.get("compiled_constraint"))
        if (
            tuple(value.get("prompt_token_ids", ())) != compiled.prompt_token_ids
            or value.get("n_low") != compiled.n_low
            or value.get("n") != compiled.n
            or value.get("nfa_states") != compiled.nfa.m
        ):
            raise ValueError(f"{job_id}: row and compiled constraint differ")
        unrolled = unroll_nfa(compiled.nfa, compiled.n, n_low=compiled.n_low)
        if sum(len(layer) for layer in unrolled.layers) != value.get("unrolled_states"):
            raise ValueError(f"{job_id}: length-unrolled state count differs")
        return cls(dict(value), compiled)

    @property
    def job_id(self) -> str:
        return str(self.source["job_id"])

    @property
    def design_index(self) -> int:
        return int(self.source["design_index"])

    def identity(self) -> dict[str, Any]:
        fields = (
            "job_id",
            "design_index",
            "family_index",
            "constraint",
            "k",
            "t",
            "n_low",
            "n",
            "nfa_states",
            "unrolled_states",
            "seed",
            "dataset",
            "structural_cell_index",
            "replicate_index",
            "base_instance_id",
            "multi_token_keyword_count",
        )
        return {
            **{field: self.source.get(field) for field in fields},
            "compiled_constraint_sha256": self.compiled.sha256,
            "job_sha256": _digest(self.source),
        }


def _model_provenance_reference(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    def reference(name: str) -> dict[str, Any]:
        value = provenance.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"workload model provenance lacks {name}")
        return {
            key: value[key]
            for key in (
                "source",
                "local",
                "resolved_path",
                "identity_sha256",
            )
            if key in value
        }

    return {
        "schema_version": 1,
        "model_profile": provenance.get("model_profile"),
        "base_model": reference("base_model"),
        "hmm_model": reference("hmm_model"),
        "model_provenance_sha256": provenance.get("sha256"),
    }


@dataclass(frozen=True)
class Workload:
    path: Path
    raw: bytes
    file_sha256: str
    dataset: str
    benchmark_name: str
    jobs_sha256: str
    model_provenance: Mapping[str, Any]
    jobs: tuple[Job, ...]

    @classmethod
    def load(cls, path: Path) -> "Workload":
        raw = path.read_bytes()
        payload = json.loads(raw)
        if not isinstance(payload, Mapping):
            raise ValueError("workload root must be an object")
        if (
            payload.get("kind") != WORKLOAD_KIND
            or payload.get("schema_version") != WORKLOAD_SCHEMA_VERSION
        ):
            raise ValueError("only current compiled dataset schema v4 is supported")
        rows = payload.get("jobs")
        if not isinstance(rows, list) or len(rows) != 500:
            raise ValueError("current dataset workload must contain exactly 500 jobs")
        if (
            payload.get("total_instances") != 500
            or payload.get("length_interval_including_eos")
            != [EXPECTED_N_LOW, EXPECTED_N]
            or payload.get("length_contract") != EXPECTED_LENGTH_CONTRACT
        ):
            raise ValueError(
                "current dataset terminal-EOS length contract differs"
            )
        supplied = payload.get("jobs_sha256")
        if not isinstance(supplied, str) or supplied != _digest(rows):
            raise ValueError("workload jobs digest mismatch")
        all_jobs = tuple(
            Job.parse(row, index) for index, row in enumerate(rows)
        )
        if any(
            (job.compiled.n_low, job.compiled.n)
            != (EXPECTED_N_LOW, EXPECTED_N)
            for job in all_jobs
        ):
            raise ValueError(
                "compiled job terminal-EOS length bounds differ"
            )
        if len({job.job_id for job in all_jobs}) != len(all_jobs):
            raise ValueError("workload job IDs are not unique")
        counts = Counter(
            str(job.source.get("constraint")) for job in all_jobs
        )
        if len(counts) != 10 or payload.get("family_counts") != dict(counts):
            raise ValueError("current workload must contain the authenticated ten families")
        dataset = payload.get("dataset")
        benchmark = payload.get("benchmark_name")
        provenance = payload.get("model_provenance")
        if (
            not isinstance(dataset, str)
            or not isinstance(benchmark, str)
            or not isinstance(provenance, Mapping)
        ):
            raise ValueError("workload metadata is incomplete")
        provenance_body = dict(provenance)
        provenance_sha256 = provenance_body.pop("sha256", None)
        if provenance_sha256 != _digest(provenance_body):
            raise ValueError("workload model provenance digest mismatch")
        for identity_name in ("base_model", "hmm_model"):
            identity = provenance_body.get(identity_name)
            if not isinstance(identity, Mapping):
                raise ValueError(
                    f"workload model provenance lacks {identity_name}"
                )
            identity_body = dict(identity)
            identity_sha256 = identity_body.pop("identity_sha256", None)
            if identity_sha256 != _digest(identity_body):
                raise ValueError(
                    f"{identity_name} provenance digest mismatch"
                )
            entries = identity_body.get("entries")
            if (
                not isinstance(entries, list)
                or identity_body.get("entries_sha256") != _digest(entries)
            ):
                raise ValueError(
                    f"{identity_name} entry manifest is invalid"
                )
        model_reference = _model_provenance_reference(provenance)
        fingerprint_sha256s = {
            _digest(job.compiled.tokenizer_fingerprint) for job in all_jobs
        }
        if len(fingerprint_sha256s) != 1:
            raise ValueError("compiled tokenizer fingerprints differ across jobs")
        for job in all_jobs:
            artifact_provenance = job.compiled.provenance
            if (
                job.source.get("dataset") != dataset
                or artifact_provenance.get("dataset") != dataset
                or artifact_provenance.get("job_id") != job.job_id
                or artifact_provenance.get("design_index") != job.design_index
                or artifact_provenance.get("seed") != job.source.get("seed")
                or artifact_provenance.get("model_provenance")
                != model_reference
            ):
                raise ValueError(
                    f"{job.job_id}: workload and artifact provenance differ"
                )
        selected_ids = set(_selected_job_ids(payload, rows))
        jobs = tuple(job for job in all_jobs if job.job_id in selected_ids)
        return cls(
            path.resolve(),
            raw,
            hashlib.sha256(raw).hexdigest(),
            dataset,
            benchmark,
            supplied,
            dict(provenance),
            jobs,
        )


@dataclass(frozen=True)
class RuntimeArtifact:
    tokenizer_info: Any
    nfa: NFA
    model_vocab_size: int
    eos_token_id: int
    terminal_token_ids: tuple[int, ...]
    token_symbol_ids: tuple[int, ...]
    symbol_token_ids: tuple[tuple[int, ...], ...]
    other_symbol_id: int
    stop_symbol_id: int
    valid_content_token_ids: tuple[int, ...]
    non_eos_special_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class GrammarMetadata:
    live_state_count: int
    rule_count: int
    edge_count: int
    initial_symbol_ids: tuple[int, ...]
    initial_allows_eos: bool


def _real_vocab_ids(tokenizer: Any, width: int) -> tuple[int, ...]:
    vocab = tokenizer.get_vocab()
    if not isinstance(vocab, Mapping) or not vocab:
        raise ValueError("tokenizer vocabulary is empty")
    result = tuple(sorted({int(value) for value in vocab.values()}))
    if result[0] < 0 or result[-1] >= width:
        raise ValueError("tokenizer IDs do not fit model logits width")
    return result


def _symbol_char(symbol: int) -> str:
    codepoint = PRIVATE_CODEPOINT_BASE + int(symbol)
    if codepoint > MAX_CODEPOINT:
        raise ValueError("too many token classes")
    return chr(codepoint)


def _symbol_terminal(symbol: int) -> str:
    return f'"\\U{PRIVATE_CODEPOINT_BASE + int(symbol):08X}"'


def _validate_tokenizer(tokenizer: Any, fingerprint: Mapping[str, Any]) -> None:
    expected = {
        "vocab_size": len(tokenizer),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "all_special_ids": [int(value) for value in tokenizer.all_special_ids],
    }
    for field, actual in expected.items():
        if fingerprint.get(field) != actual:
            raise ValueError(f"runtime tokenizer fingerprint differs: {field}")


def _configured_terminal_token_ids(
    tokenizer: Any,
    config: Any,
    generation_config: Any,
    *,
    model_vocab_size: int,
) -> tuple[int, ...]:
    """Collect the model-native STOP aliases used by the other decoders."""

    if tokenizer.eos_token_id is None:
        raise ValueError("target tokenizer must define EOS")
    values: list[int] = [int(tokenizer.eos_token_id)]
    sources = (
        getattr(tokenizer, "eos_token_ids", None),
        getattr(config, "eos_token_id", None),
        getattr(getattr(config, "text_config", None), "eos_token_id", None),
        getattr(generation_config, "eos_token_id", None),
    )
    for source in sources:
        if source is None:
            continue
        candidates = (
            source if isinstance(source, (list, tuple, set)) else (source,)
        )
        for candidate in candidates:
            if isinstance(candidate, bool):
                raise ValueError("terminal token IDs cannot be booleans")
            token_id = int(candidate)
            if not 0 <= token_id < model_vocab_size:
                raise ValueError(
                    f"configured terminal token_id={token_id} is outside the "
                    "model vocabulary"
                )
            values.append(token_id)
    terminal_ids = tuple(dict.fromkeys(values))
    special_ids = {
        int(token_id) for token_id in getattr(tokenizer, "all_special_ids", ())
    }
    if any(token_id not in special_ids for token_id in terminal_ids):
        raise ValueError("model terminal IDs must be tokenizer special tokens")
    return terminal_ids


def _load_terminal_policy(
    model_path: Path,
    *,
    trust_remote_code: bool,
    local_files_only: bool,
) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    try:
        generation_config = GenerationConfig.from_pretrained(
            model_path,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
    except OSError:
        generation_config = GenerationConfig.from_model_config(config)
    model_vocab_size = _config_vocab_size(config)
    terminal_ids = _configured_terminal_token_ids(
        tokenizer,
        config,
        generation_config,
        model_vocab_size=model_vocab_size,
    )
    return {
        "canonical_eos_token_id": int(tokenizer.eos_token_id),
        "terminal_token_ids": list(terminal_ids),
        "terminal_token_texts": [
            str(tokenizer.convert_ids_to_tokens(token_id))
            for token_id in terminal_ids
        ],
        "policy": (
            "all model-native terminal aliases are one STOP class; terminal "
            "identity is excluded from content"
        ),
    }


def token_nfa_from_compiled(
    tokenizer: Any,
    model_vocab_size: int,
    compiled: CompiledConstraint,
    terminal_token_ids: Sequence[int] | None = None,
) -> RuntimeArtifact:
    _validate_tokenizer(tokenizer, compiled.tokenizer_fingerprint)
    eos = _integer(tokenizer.eos_token_id, "eos_token_id")
    terminals = tuple(
        dict.fromkeys(
            _integer(token_id, "terminal token ID")
            for token_id in (
                (eos,) if terminal_token_ids is None else terminal_token_ids
            )
        )
    )
    if not terminals or terminals[0] != eos:
        raise ValueError("terminal aliases must begin with canonical EOS")
    real = _real_vocab_ids(tokenizer, model_vocab_size)
    real_set = set(real)
    special = {
        int(value)
        for value in tokenizer.all_special_ids
        if 0 <= int(value) < model_vocab_size
    }
    if any(token not in real_set or token not in special for token in terminals):
        raise ValueError(
            "terminal aliases must be real tokenizer special tokens"
        )
    valid = tuple(token for token in real if token not in special)
    valid_set = set(valid)
    explicit: dict[int, int] = {}
    for token_class in compiled.token_classes:
        for token in token_class.token_ids:
            if token not in valid_set:
                raise ValueError("compiled class contains a non-content token")
            explicit[token] = token_class.symbol_id
    other_ids = tuple(token for token in valid if token not in explicit)
    symbols = [-1] * model_vocab_size
    for token in other_ids:
        symbols[token] = compiled.other_symbol_id
    for token, symbol in explicit.items():
        symbols[token] = symbol
    for terminal in terminals:
        symbols[terminal] = compiled.stop_symbol_id
    symbol_tokens = (
        tuple(token_class.token_ids for token_class in compiled.token_classes)
        + (other_ids, terminals)
    )
    encoded_vocab = [""] * (max(real) + 1)
    terminal_set = set(terminals)
    for token in real:
        symbol = symbols[token]
        if token not in terminal_set and symbol >= 0:
            encoded_vocab[token] = _symbol_char(symbol)
    tokenizer_info = xgr.TokenizerInfo(
        encoded_vocab,
        vocab_size=model_vocab_size,
        stop_token_ids=list(terminals),
    )
    return RuntimeArtifact(
        tokenizer_info,
        compiled.nfa,
        model_vocab_size,
        eos,
        terminals,
        tuple(symbols),
        symbol_tokens,
        compiled.other_symbol_id,
        compiled.stop_symbol_id,
        valid,
        tuple(sorted(special - set(terminals))),
    )


def _successors(nfa: NFA, state: Any, symbol: int) -> set[Any]:
    result = set(nfa.trans[state].get(symbol, ()))
    result.update(nfa.any_trans.get(state, ()))
    return result


def nfa_to_bounded_ebnf(
    artifact: RuntimeArtifact, n_low: int, n: int
) -> tuple[str, GrammarMetadata]:
    min_content, max_content = n_low - 1, n - 1
    if len(artifact.nfa.initials) != 1:
        raise ValueError("bounded grammar requires one NFA initial state")
    nfa = artifact.nfa
    content_symbols = tuple(
        symbol
        for symbol in range(artifact.stop_symbol_id)
        if artifact.symbol_token_ids[symbol]
    )
    reachable: list[set[Any]] = [set() for _ in range(max_content + 1)]
    reachable[0].add(nfa.initials[0])
    all_edges: dict[tuple[int, Any], tuple[tuple[int, Any], ...]] = {}
    for layer in range(max_content):
        for state in sorted(reachable[layer], key=nfa.rank):
            edges = {
                (symbol, destination)
                for symbol in content_symbols
                for destination in _successors(nfa, state, symbol)
            }
            all_edges[layer, state] = tuple(
                sorted(edges, key=lambda item: (item[0], nfa.rank(item[1])))
            )
            reachable[layer + 1].update(destination for _, destination in edges)
    finals = set(nfa.finals)
    accepting = {
        (layer, state)
        for layer in range(min_content, max_content + 1)
        for state in reachable[layer]
        if _successors(nfa, state, artifact.stop_symbol_id) & finals
    }
    live = set(accepting)
    for layer in range(max_content - 1, -1, -1):
        live.update(
            (layer, state)
            for state in reachable[layer]
            if any(
                (layer + 1, destination) in live
                for _, destination in all_edges.get((layer, state), ())
            )
        )
    root = (0, nfa.initials[0])
    if root not in live:
        raise ValueError("compiled constraint has no accepted bounded word")
    state_index = {state: index for index, state in enumerate(nfa.states)}

    def rule(layer: int, state: Any) -> str:
        return f"q_{layer}_{state_index[state]}"

    transition_edges: list[tuple[int, Any, int, Any]] = []
    used_symbols: set[int] = set()
    for layer in range(max_content):
        for state in sorted(
            (state for candidate, state in live if candidate == layer), key=nfa.rank
        ):
            for symbol, destination in all_edges.get((layer, state), ()):
                if (layer + 1, destination) in live:
                    transition_edges.append((layer, state, symbol, destination))
                    used_symbols.add(symbol)
    lines = [f"root ::= {rule(*root)}"]
    lines.extend(
        f"tok_{symbol} ::= {_symbol_terminal(symbol)}"
        for symbol in sorted(used_symbols)
    )
    initial_symbols: set[int] = set()
    edge_count = 0
    for layer in range(max_content + 1):
        for state in sorted(
            (state for candidate, state in live if candidate == layer), key=nfa.rank
        ):
            branches: list[str] = []
            for symbol, destination in all_edges.get((layer, state), ()):
                if (layer + 1, destination) in live:
                    branches.append(f"tok_{symbol} {rule(layer + 1, destination)}")
                    edge_count += 1
                    if layer == 0 and state == root[1]:
                        initial_symbols.add(symbol)
            if (layer, state) in accepting:
                branches.append('""')
                edge_count += 1
            lines.append(f"{rule(layer, state)} ::= " + " | ".join(branches))
    metadata = GrammarMetadata(
        len(live),
        1 + len(used_symbols) + len(live),
        edge_count,
        tuple(sorted(initial_symbols)),
        root in accepting,
    )
    return "\n".join(lines) + "\n", metadata


def audit_initial_mask(
    compiled_grammar: Any,
    artifact: RuntimeArtifact,
    metadata: GrammarMetadata,
) -> dict[str, Any]:
    matcher = xgr.GrammarMatcher(compiled_grammar)
    bitmask = xgr.allocate_token_bitmask(1, artifact.model_vocab_size)
    matcher.fill_next_token_bitmask(bitmask)
    logits = torch.zeros((1, artifact.model_vocab_size), dtype=torch.float32)
    xgr.apply_token_bitmask_inplace(
        logits, bitmask, vocab_size=artifact.model_vocab_size, backend="torch_native"
    )
    actual = torch.isfinite(logits[0])
    expected = torch.zeros((artifact.model_vocab_size,), dtype=torch.bool)
    for symbol in metadata.initial_symbol_ids:
        for token in artifact.symbol_token_ids[symbol]:
            expected[token] = True
    if metadata.initial_allows_eos:
        for terminal in artifact.terminal_token_ids:
            expected[terminal] = True
    missing = (expected & ~actual).nonzero().flatten().tolist()
    unexpected = (actual & ~expected).nonzero().flatten().tolist()
    stops = sorted(int(value) for value in matcher.stop_token_ids)
    ok = (
        not missing
        and not unexpected
        and stops == sorted(artifact.terminal_token_ids)
    )
    result = {
        "ok": ok,
        "allowed_count": int(actual.sum()),
        "expected_count": int(expected.sum()),
        "missing_expected": missing,
        "unexpected_allowed": unexpected,
        "stop_token_ids": stops,
    }
    if not ok:
        raise AssertionError(f"XGrammar initial mask differs from NFA oracle: {result}")
    return result


def accepts_generation(
    artifact: RuntimeArtifact, token_ids: Sequence[int], n_low: int, n: int
) -> bool:
    ids = tuple(int(value) for value in token_ids)
    terminals = set(artifact.terminal_token_ids)
    if (
        not n_low <= len(ids) <= n
        or not ids
        or ids[-1] not in terminals
        or any(token in terminals for token in ids[:-1])
    ):
        return False
    active = set(artifact.nfa.initials)
    for token in ids:
        if not 0 <= token < artifact.model_vocab_size:
            return False
        symbol = artifact.token_symbol_ids[token]
        if symbol < 0:
            return False
        active = {
            destination
            for state in active
            for destination in _successors(artifact.nfa, state, symbol)
        }
        if not active:
            return False
    return bool(active & set(artifact.nfa.finals))


def _torch_dtype(name: str) -> Any:
    if name == "auto":
        return "auto"
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _model_vocab_size(model: Any) -> int:
    embeddings = model.get_input_embeddings()
    if embeddings is None:
        raise ValueError("model has no input embeddings")
    return int(embeddings.num_embeddings)


def _config_vocab_size(config: Any) -> int:
    text_config = getattr(config, "text_config", None)
    value = getattr(text_config, "vocab_size", None)
    if value is None:
        value = getattr(config, "vocab_size", None)
    width = _integer(value, "model config vocab_size")
    if width <= 0:
        raise ValueError("model config vocab_size must be positive")
    return width


@torch.inference_mode()
def _generate(
    model: Any,
    tokenizer: Any,
    prompt_ids: Sequence[int],
    compiled_grammar: Any,
    *,
    n: int,
    terminal_token_ids: Sequence[int],
    temperature: float,
    top_p: float,
    top_k: int,
    device: torch.device,
) -> tuple[list[int], float]:
    inputs = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    processor = xgr.contrib.hf.LogitsProcessor(compiled_grammar)
    terminals = [int(token_id) for token_id in terminal_token_ids]
    if not terminals:
        raise ValueError("generation requires at least one terminal token")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    output = model.generate(
        input_ids=inputs,
        attention_mask=torch.ones_like(inputs),
        do_sample=True,
        max_new_tokens=n,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        logits_processor=[processor],
        eos_token_id=terminals[0] if len(terminals) == 1 else terminals,
        pad_token_id=int(tokenizer.pad_token_id),
        use_cache=True,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return (
        [int(value) for value in output[0, inputs.shape[1] :].cpu().tolist()],
        time.perf_counter() - started,
    )


def _worker(
    job: Job,
    args: argparse.Namespace,
    progress_path: Path,
) -> dict[str, Any]:
    device = torch.device(args.device)
    tokenizer_started = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    _validate_tokenizer(tokenizer, job.compiled.tokenizer_fingerprint)
    config = AutoConfig.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    try:
        generation_config = GenerationConfig.from_pretrained(
            args.base_model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
    except OSError:
        generation_config = GenerationConfig.from_model_config(config)
    terminal_ids = _configured_terminal_token_ids(
        tokenizer,
        config,
        generation_config,
        model_vocab_size=_config_vocab_size(config),
    )
    expected_terminal_ids = tuple(
        _integer(value, "expected terminal token ID")
        for value in args.terminal_token_ids.split(",")
        if value
    )
    if terminal_ids != expected_terminal_ids:
        raise ValueError(
            "worker model-native terminal aliases differ from launch contract"
        )
    tokenizer_config_load_s = time.perf_counter() - tokenizer_started
    seed = int(job.source["seed"])
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(device)
    build_started = time.perf_counter()
    artifact = token_nfa_from_compiled(
        tokenizer,
        _config_vocab_size(config),
        job.compiled,
        terminal_ids,
    )
    ebnf, metadata = nfa_to_bounded_ebnf(
        artifact, job.compiled.n_low, job.compiled.n
    )
    grammar = xgr.Grammar.from_ebnf(ebnf)
    compiler = xgr.GrammarCompiler(
        artifact.tokenizer_info,
        max_threads=max(1, args.cpu_threads),
        cache_enabled=False,
    )
    grammar_build_s = time.perf_counter() - build_started
    compile_started = time.perf_counter()
    compiled_grammar = compiler.compile_grammar(grammar)
    compile_s = time.perf_counter() - compile_started
    precompute_s = grammar_build_s + compile_s
    precompute_metrics = {
        "grammar_build_runtime_s": grammar_build_s,
        "compile_runtime_s": compile_s,
        "xgrammar_precompute_runtime_s": precompute_s,
        "setup_runtime_s": precompute_s,
        "grammar_rule_count": metadata.rule_count,
        "grammar_edge_count": metadata.edge_count,
        "grammar_bytes": len(ebnf.encode()),
        "compiled_grammar_memory_bytes": int(
            compiled_grammar.memory_size_bytes
        ),
        "model_vocab_size": artifact.model_vocab_size,
    }
    _write_json(
        progress_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            **job.identity(),
            "phase": "precompute_complete",
            "metrics": precompute_metrics,
        },
    )
    mask_audit = audit_initial_mask(compiled_grammar, artifact, metadata)
    _write_json(
        progress_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            **job.identity(),
            "phase": "mask_audit_complete",
            "metrics": {
                **precompute_metrics,
                "initial_mask_audit": mask_audit,
            },
        },
    )
    model_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=_torch_dtype(args.dtype),
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    ).to(device).eval()
    model_load_s = time.perf_counter() - model_started
    if _model_vocab_size(model) != artifact.model_vocab_size:
        raise ValueError(
            "loaded model vocabulary differs from authenticated config"
        )
    _write_json(
        progress_path,
        {
            "schema_version": RESULT_SCHEMA_VERSION,
            **job.identity(),
            "phase": "model_loaded",
            "metrics": {
                **precompute_metrics,
                "initial_mask_audit": mask_audit,
                "model_load_runtime_s": model_load_s,
            },
        },
    )
    generated, generation_s = _generate(
        model,
        tokenizer,
        job.compiled.prompt_token_ids,
        compiled_grammar,
        n=job.compiled.n,
        terminal_token_ids=artifact.terminal_token_ids,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
    )
    if not accepts_generation(
        artifact, generated, job.compiled.n_low, job.compiled.n
    ):
        raise RuntimeError("XGrammar generated outside the compiled token NFA")
    content_ids = generated[:-1]
    keyword_occurrence_counts = [
        _overlapping_occurrence_count(content_ids, pattern)
        for pattern in job.source["tracked_keyword_token_ids"]
    ]
    return {
        **job.identity(),
        "method": METHOD,
        "status": "success",
        "failure_kind": None,
        "tokenizer_config_load_runtime_s": tokenizer_config_load_s,
        "model_load_runtime_s": model_load_s,
        **precompute_metrics,
        "generation_runtime_s": generation_s,
        "prompt_token_count": len(job.compiled.prompt_token_ids),
        "generated_total_len": len(generated),
        "generated_content_len": len(content_ids),
        "generated_token_ids": generated,
        "generated_content_token_ids": content_ids,
        "canonical_eos_token_id": int(artifact.eos_token_id),
        "terminal_token_ids": list(artifact.terminal_token_ids),
        "terminal_eos_token_id": int(generated[-1]),
        "terminal_eos_count": 1,
        "terminated_with_eos": True,
        "hit_content_token_cap": len(content_ids) == job.compiled.n - 1,
        "keyword_occurrence_counts": keyword_occurrence_counts,
        "generated_text": tokenizer.decode(
            content_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        ),
        "valid_generation": True,
        "initial_mask_audit": mask_audit,
        "peak_gpu_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
    }


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--job_json", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--base_model_path", required=True)
    parser.add_argument("--terminal_token_ids", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--cpu_threads", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    return parser


def _worker_main(argv: Sequence[str]) -> int:
    args = _worker_parser().parse_args(argv)
    source = json.loads(args.job_json.read_text(encoding="utf-8"))
    job = Job.parse(source, int(source["design_index"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    started = time.perf_counter()
    result = _worker(job, args, progress_path)
    result["worker_total_runtime_s"] = time.perf_counter() - started
    _write_json(args.output_dir / "results.json", {"complete": True, "result": result})
    return 0


def _register(pid: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.add(pid)


def _unregister(pid: int) -> None:
    with _ACTIVE_LOCK:
        _ACTIVE.discard(pid)


def _terminate(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _terminate_all() -> None:
    with _ACTIVE_LOCK:
        active = tuple(_ACTIVE)
    for pid in active:
        _terminate(pid)


def _tail(path: Path, size: int = 16000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        length = handle.tell()
        handle.seek(max(0, length - size))
        return handle.read().decode(errors="replace")


def _next_attempt(job_dir: Path) -> tuple[int, Path]:
    attempts = job_dir / "attempts"
    attempts.mkdir(parents=True, exist_ok=True)
    existing = [
        int(path.name.rsplit("_", 1)[1])
        for path in attempts.glob("attempt_*")
        if path.name.rsplit("_", 1)[-1].isdigit()
    ]
    number = max(existing, default=0) + 1
    path = attempts / f"attempt_{number:03d}"
    path.mkdir()
    return number, path


def _validate_worker_result(
    job: Job,
    result: Mapping[str, Any],
    terminal_token_ids: Sequence[int] | None = None,
) -> dict[str, Any]:
    normalized = dict(result)
    for field, expected in job.identity().items():
        if normalized.get(field) != expected:
            raise ValueError(f"worker result changed {field}")
    if (
        normalized.get("method") != METHOD
        or normalized.get("status") != "success"
        or normalized.get("failure_kind") is not None
        or normalized.get("valid_generation") is not True
        or normalized.get("prompt_token_count")
        != len(job.compiled.prompt_token_ids)
    ):
        raise ValueError("worker result completion contract is invalid")
    precompute = _runtime_metric(
        normalized,
        "xgrammar_precompute_runtime_s",
        positive=True,
    )
    setup = _runtime_metric(normalized, "setup_runtime_s", positive=True)
    grammar_build = _runtime_metric(
        normalized,
        "grammar_build_runtime_s",
        positive=True,
    )
    compile_runtime = _runtime_metric(
        normalized,
        "compile_runtime_s",
        positive=True,
    )
    _runtime_metric(normalized, "tokenizer_config_load_runtime_s")
    _runtime_metric(normalized, "model_load_runtime_s")
    _runtime_metric(normalized, "generation_runtime_s")
    _runtime_metric(normalized, "worker_total_runtime_s", positive=True)
    if (
        setup != precompute
        or not math.isclose(
            precompute,
            grammar_build + compile_runtime,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
        or any(
            isinstance(normalized.get(field), bool)
            or not isinstance(normalized.get(field), int)
            or int(normalized[field]) <= 0
            for field in (
                "grammar_rule_count",
                "grammar_edge_count",
                "grammar_bytes",
                "compiled_grammar_memory_bytes",
                "model_vocab_size",
            )
        )
        or not isinstance(normalized.get("initial_mask_audit"), Mapping)
        or normalized["initial_mask_audit"].get("ok") is not True
    ):
        raise ValueError("worker result runtime contract is invalid")
    generated = normalized.get("generated_token_ids")
    expected_terminals = tuple(
        int(token_id)
        for token_id in (
            (
                job.compiled.tokenizer_fingerprint.get("eos_token_id"),
            )
            if terminal_token_ids is None
            else terminal_token_ids
        )
    )
    validation_artifact = token_nfa_from_compiled_for_validation(
        job.compiled,
        expected_terminals,
    )
    content = generated[:-1] if isinstance(generated, list) else None
    expected_occurrence_counts = (
        [
            _overlapping_occurrence_count(content, pattern)
            for pattern in job.source["tracked_keyword_token_ids"]
        ]
        if content is not None
        else None
    )
    if (
        not isinstance(generated, list)
        or any(
            isinstance(token, bool)
            or not isinstance(token, int)
            or token < 0
            or token >= validation_artifact.model_vocab_size
            for token in generated
        )
        or not accepts_generation(
            validation_artifact,
            generated,
            job.compiled.n_low,
            job.compiled.n,
        )
        or normalized.get("generated_total_len") != len(generated)
        or normalized.get("generated_content_len") != len(generated) - 1
        or normalized.get("generated_content_token_ids") != generated[:-1]
        or normalized.get("canonical_eos_token_id")
        != validation_artifact.eos_token_id
        or normalized.get("terminal_token_ids")
        != list(validation_artifact.terminal_token_ids)
        or normalized.get("terminal_eos_token_id")
        != generated[-1]
        or normalized.get("terminal_eos_count") != 1
        or normalized.get("terminated_with_eos") is not True
        or normalized.get("hit_content_token_cap")
        is not (len(generated) - 1 == job.compiled.n - 1)
        or normalized.get("keyword_occurrence_counts")
        != expected_occurrence_counts
        or not isinstance(normalized.get("generated_text"), str)
    ):
        raise ValueError("worker result token sequence is invalid")
    return normalized


def token_nfa_from_compiled_for_validation(
    compiled: CompiledConstraint,
    terminal_token_ids: Sequence[int] | None = None,
) -> RuntimeArtifact:
    """Return the token-symbol portion needed to recheck a worker sequence."""

    max_token = max(
        token
        for token_class in compiled.token_classes
        for token in token_class.token_ids
    )
    eos = _integer(
        compiled.tokenizer_fingerprint.get("eos_token_id"),
        "compiled eos_token_id",
    )
    terminals = tuple(
        dict.fromkeys(
            _integer(token_id, "terminal token ID")
            for token_id in (
                (eos,) if terminal_token_ids is None else terminal_token_ids
            )
        )
    )
    if not terminals or terminals[0] != eos:
        raise ValueError("terminal aliases must begin with canonical EOS")
    tokenizer_size = _integer(
        compiled.tokenizer_fingerprint.get("vocab_size"),
        "compiled tokenizer vocab_size",
    )
    width = max(tokenizer_size, max_token + 1, max(terminals) + 1)
    symbols = [compiled.other_symbol_id] * width
    specials = {
        _integer(token, "compiled special token ID")
        for token in compiled.tokenizer_fingerprint.get("all_special_ids", ())
    }
    if any(token not in specials or not 0 <= token < width for token in terminals):
        raise ValueError(
            "terminal aliases must be compiled tokenizer special tokens"
        )
    for special in specials:
        if 0 <= special < width:
            symbols[special] = -1
    for token_class in compiled.token_classes:
        for token in token_class.token_ids:
            symbols[token] = token_class.symbol_id
    for terminal in terminals:
        symbols[terminal] = compiled.stop_symbol_id
    return RuntimeArtifact(
        tokenizer_info=None,
        nfa=compiled.nfa,
        model_vocab_size=width,
        eos_token_id=eos,
        terminal_token_ids=terminals,
        token_symbol_ids=tuple(symbols),
        symbol_token_ids=tuple(
            token_class.token_ids for token_class in compiled.token_classes
        )
        + ((), terminals),
        other_symbol_id=compiled.other_symbol_id,
        stop_symbol_id=compiled.stop_symbol_id,
        valid_content_token_ids=(),
        non_eos_special_token_ids=(),
    )


def _runtime_metric(
    metrics: Mapping[str, Any],
    name: str,
    *,
    positive: bool = False,
) -> float:
    value = metrics.get(name)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (float(value) <= 0 if positive else float(value) < 0)
    ):
        qualifier = "positive" if positive else "nonnegative"
        raise ValueError(f"progress metric {name} must be {qualifier}")
    return float(value)


def _validate_progress(job: Job, payload: Mapping[str, Any]) -> dict[str, Any]:
    progress = dict(payload)
    if progress.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("worker progress schema differs")
    for field, expected in job.identity().items():
        if progress.get(field) != expected:
            raise ValueError(f"worker progress changed {field}")
    phase = progress.get("phase")
    if not isinstance(phase, str) or phase not in _PROGRESS_PHASES:
        raise ValueError("worker progress phase is invalid")
    metrics = progress.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("worker progress metrics are missing")
    normalized_metrics = dict(metrics)
    precompute = _runtime_metric(
        normalized_metrics,
        "xgrammar_precompute_runtime_s",
        positive=True,
    )
    setup = _runtime_metric(
        normalized_metrics,
        "setup_runtime_s",
        positive=True,
    )
    if setup != precompute:
        raise ValueError("worker progress setup runtime differs")
    grammar_build = _runtime_metric(
        normalized_metrics,
        "grammar_build_runtime_s",
        positive=True,
    )
    compile_runtime = _runtime_metric(
        normalized_metrics,
        "compile_runtime_s",
        positive=True,
    )
    if not math.isclose(
        precompute,
        grammar_build + compile_runtime,
        rel_tol=1e-12,
        abs_tol=1e-12,
    ):
        raise ValueError("worker progress precompute components differ")
    if _PROGRESS_PHASES[str(phase)] >= _PROGRESS_PHASES["mask_audit_complete"]:
        audit = normalized_metrics.get("initial_mask_audit")
        if not isinstance(audit, Mapping) or audit.get("ok") is not True:
            raise ValueError("worker progress mask audit is invalid")
    if phase == "model_loaded":
        _runtime_metric(normalized_metrics, "model_load_runtime_s")
    progress["metrics"] = normalized_metrics
    return progress


def _read_progress(job: Job, path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("worker progress must be an object")
    return _validate_progress(job, payload)


def _crosscheck_progress_result(
    progress: Mapping[str, Any],
    result: Mapping[str, Any],
) -> None:
    phase = str(progress["phase"])
    metrics = progress["metrics"]
    fields = [
        "grammar_build_runtime_s",
        "compile_runtime_s",
        "xgrammar_precompute_runtime_s",
        "setup_runtime_s",
        "grammar_rule_count",
        "grammar_edge_count",
        "grammar_bytes",
        "compiled_grammar_memory_bytes",
        "model_vocab_size",
    ]
    if _PROGRESS_PHASES[phase] >= _PROGRESS_PHASES["mask_audit_complete"]:
        fields.append("initial_mask_audit")
    if phase == "model_loaded":
        fields.append("model_load_runtime_s")
    changed = [
        field for field in fields if metrics.get(field) != result.get(field)
    ]
    if changed:
        raise ValueError(
            "worker result differs from progress checkpoint: "
            + ", ".join(changed)
        )


def _run_one(
    args: argparse.Namespace,
    run_dir: Path,
    job: Job,
    gpu_pool: "queue.Queue[str]",
    run_contract_sha256: str,
) -> dict[str, Any]:
    job_dir = run_dir / "jobs" / job.job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    attempt, attempt_dir = _next_attempt(job_dir)
    job_json = attempt_dir / "job.json"
    output_dir = attempt_dir / "worker_output"
    progress_path = output_dir / "progress.json"
    log_path = attempt_dir / "run.log"
    _write_json(job_json, job.source)
    command = [
        str(args.python),
        str(SCRIPT),
        "--internal-worker",
        "--job_json",
        str(job_json),
        "--output_dir",
        str(output_dir),
        "--base_model_path",
        args.base_model_path,
        "--terminal_token_ids",
        ",".join(str(token_id) for token_id in args.terminal_token_ids),
        "--dtype",
        args.dtype,
        "--cpu_threads",
        str(args.cpu_threads_per_worker),
        "--temperature",
        str(args.temperature),
        "--top_p",
        str(args.top_p),
        "--top_k",
        str(args.top_k),
    ]
    if args.trust_remote_code:
        command.append("--trust_remote_code")
    if args.local_files_only:
        command.append("--local_files_only")
    gpu = gpu_pool.get()
    cuda_visible_device = _worker_cuda_device(gpu)
    started = time.perf_counter()
    timed_out = False
    timeout_phase: str | None = None
    returncode: int | None = None
    progress: dict[str, Any] | None = None
    progress_error: str | None = None
    try:
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": cuda_visible_device,
            "OMP_NUM_THREADS": str(args.cpu_threads_per_worker),
            "MKL_NUM_THREADS": str(args.cpu_threads_per_worker),
            "TOKENIZERS_PARALLELISM": "false",
        }
        with log_path.open("w", encoding="utf-8") as log:
            log.write("command: " + shlex.join(command) + "\n")
            log.flush()
            process = subprocess.Popen(
                command,
                cwd=XGRAMMAR_ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            _register(process.pid)
            try:
                precompute_deadline = (
                    time.monotonic() + args.precompute_timeout_s
                )
                post_precompute_deadline: float | None = None
                while returncode is None:
                    try:
                        observed = _read_progress(job, progress_path)
                    except (
                        OSError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as error:
                        progress_error = str(error)
                        _terminate(process.pid)
                        returncode = process.wait()
                        break
                    if observed is not None:
                        if (
                            progress is not None
                            and _PROGRESS_PHASES[str(observed["phase"])]
                            < _PROGRESS_PHASES[str(progress["phase"])]
                        ):
                            progress_error = "worker progress phase regressed"
                            _terminate(process.pid)
                            returncode = process.wait()
                            break
                        progress = observed
                        if post_precompute_deadline is None:
                            post_precompute_deadline = (
                                time.monotonic()
                                + args.generation_timeout_s
                            )
                    deadline = (
                        post_precompute_deadline
                        if post_precompute_deadline is not None
                        else precompute_deadline
                    )
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        timed_out = True
                        timeout_phase = (
                            "post_precompute"
                            if post_precompute_deadline is not None
                            else "precompute"
                        )
                        _terminate(process.pid)
                        returncode = process.wait()
                        break
                    try:
                        returncode = process.wait(
                            timeout=min(0.25, remaining)
                        )
                    except subprocess.TimeoutExpired:
                        pass
                if progress_error is None:
                    try:
                        observed = _read_progress(job, progress_path)
                        if observed is not None:
                            progress = observed
                    except (
                        OSError,
                        TypeError,
                        ValueError,
                        json.JSONDecodeError,
                    ) as error:
                        progress_error = str(error)
            finally:
                _unregister(process.pid)
    finally:
        gpu_pool.put(gpu)
    result_path = output_dir / "results.json"
    result: dict[str, Any] | None = None
    result_error: str | None = None
    if progress is None and returncode == 0 and not timed_out:
        progress_error = progress_error or "worker progress is missing"
    if (
        returncode == 0
        and not timed_out
        and progress_error is None
        and result_path.is_file()
    ):
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("complete") is not True or not isinstance(
                payload.get("result"), Mapping
            ):
                raise ValueError("worker result envelope is incomplete")
            result = _validate_worker_result(
                job,
                payload["result"],
                args.terminal_token_ids,
            )
            if progress is None:
                raise ValueError("worker result lacks progress checkpoint")
            _crosscheck_progress_result(progress, result)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
            result_error = str(error)
    success = result is not None
    log_tail = _tail(log_path)
    failure_kind = None
    if not success:
        lower = log_tail.lower()
        failure_kind = (
            f"{timeout_phase}_timeout"
            if timed_out
            else "invalid_progress"
            if progress_error is not None
            else "cuda_oom"
            if "out of memory" in lower
            else "invalid_result"
            if returncode == 0
            else "worker_error"
        )
    status = {
        **job.identity(),
        "run_contract_sha256": run_contract_sha256,
        "status": "success" if success else "failed",
        "failure_kind": failure_kind,
        "timeout_s": args.timeout_s,
        "precompute_timeout_s": args.precompute_timeout_s,
        "generation_timeout_s": args.generation_timeout_s,
        "timed_out": timed_out,
        "timeout_phase": timeout_phase,
        "attempt": attempt,
        "worker_gpu": gpu,
        "cuda_visible_device": cuda_visible_device,
        "worker_subprocess_runtime_s": time.perf_counter() - started,
        "progress_phase": progress.get("phase") if progress else None,
        "progress_metrics": progress.get("metrics") if progress else None,
        "result": result,
        "log_path": str(log_path),
        "result_path": str(result_path) if result_path.is_file() else None,
    }
    if not success:
        status["log_tail"] = log_tail
        status["progress_error"] = progress_error
        status["result_error"] = result_error
    _write_json(job_dir / "status.json", status)
    return status


def _load_statuses(
    run_dir: Path,
    jobs: Sequence[Job],
    run_contract_sha256: str,
    expected_timeouts: Mapping[str, float] | None = None,
    terminal_token_ids: Sequence[int] | None = None,
) -> dict[str, dict[str, Any]]:
    statuses: dict[str, dict[str, Any]] = {}
    for job in jobs:
        path = run_dir / "jobs" / job.job_id / "status.json"
        if not path.is_file():
            continue
        status = json.loads(path.read_text(encoding="utf-8"))
        if status.get("job_sha256") != _digest(job.source):
            raise ValueError(f"{job.job_id}: saved status belongs to another job")
        if status.get("run_contract_sha256") != run_contract_sha256:
            raise ValueError(
                f"{job.job_id}: saved status belongs to another run contract"
            )
        terminal = status.get("status")
        if terminal not in {"success", "failed"}:
            raise ValueError(f"{job.job_id}: saved status is not terminal")
        for field, expected in job.identity().items():
            if status.get(field) != expected:
                raise ValueError(f"{job.job_id}: saved status changed {field}")
        for field in (
            "timeout_s",
            "precompute_timeout_s",
            "generation_timeout_s",
        ):
            _runtime_metric(status, field, positive=True)
            if (
                expected_timeouts is not None
                and status.get(field) != expected_timeouts[field]
            ):
                raise ValueError(
                    f"{job.job_id}: saved status changed {field}"
                )
        progress_phase = status.get("progress_phase")
        progress_metrics = status.get("progress_metrics")
        if progress_phase is None and progress_metrics is None:
            progress = None
        elif progress_phase is None or not isinstance(
            progress_metrics, Mapping
        ):
            raise ValueError(
                f"{job.job_id}: saved progress checkpoint is incomplete"
            )
        else:
            progress = _validate_progress(
                job,
                {
                    "schema_version": RESULT_SCHEMA_VERSION,
                    **job.identity(),
                    "phase": progress_phase,
                    "metrics": progress_metrics,
                },
            )
            status["progress_metrics"] = progress["metrics"]
        if terminal == "success":
            result = status.get("result")
            if (
                not isinstance(result, Mapping)
                or status.get("failure_kind") is not None
                or status.get("timed_out") is not False
                or status.get("timeout_phase") is not None
                or progress is None
                or progress.get("phase") != "model_loaded"
            ):
                raise ValueError(
                    f"{job.job_id}: saved success result is incomplete"
                )
            status["result"] = _validate_worker_result(
                job,
                result,
                terminal_token_ids,
            )
            _crosscheck_progress_result(progress, status["result"])
        elif (
            status.get("result") is not None
            or status.get("failure_kind") not in _FAILURE_KINDS
        ):
            raise ValueError(f"{job.job_id}: saved failure result is invalid")
        else:
            failure_kind = str(status["failure_kind"])
            if failure_kind == "precompute_timeout":
                coherent = (
                    status.get("timed_out") is True
                    and status.get("timeout_phase") == "precompute"
                    and progress is None
                )
            elif failure_kind == "post_precompute_timeout":
                coherent = (
                    status.get("timed_out") is True
                    and status.get("timeout_phase") == "post_precompute"
                    and progress is not None
                )
            else:
                coherent = (
                    status.get("timed_out") is False
                    and status.get("timeout_phase") is None
                )
            if not coherent:
                raise ValueError(
                    f"{job.job_id}: saved failure timeout is inconsistent"
                )
        statuses[job.job_id] = status
    return statuses


def _flat_row(job: Job, status: Mapping[str, Any] | None) -> dict[str, Any]:
    if status is None:
        return {
            **job.identity(),
            "status": "pending",
            "failure_kind": None,
        }
    progress_metrics = status.get("progress_metrics")
    metrics = (
        dict(progress_metrics)
        if isinstance(progress_metrics, Mapping)
        else {}
    )
    result = status.get("result")
    if isinstance(result, Mapping):
        metrics.update(result)
    for field in job.identity():
        metrics.pop(field, None)
    metrics.pop("status", None)
    metrics.pop("failure_kind", None)
    return {
        **job.identity(),
        "status": status["status"],
        "failure_kind": status.get("failure_kind"),
        "timeout_s": status.get("timeout_s"),
        "precompute_timeout_s": status.get("precompute_timeout_s"),
        "generation_timeout_s": status.get("generation_timeout_s"),
        "timed_out": status.get("timed_out"),
        "timeout_phase": status.get("timeout_phase"),
        "progress_phase": status.get("progress_phase"),
        "worker_subprocess_runtime_s": status.get("worker_subprocess_runtime_s"),
        **metrics,
    }


def _launcher_error_status(
    args: argparse.Namespace,
    run_dir: Path,
    job: Job,
    run_contract_sha256: str,
    error: BaseException,
) -> dict[str, Any]:
    status = {
        **job.identity(),
        "run_contract_sha256": run_contract_sha256,
        "status": "failed",
        "failure_kind": "launcher_error",
        "timeout_s": args.timeout_s,
        "precompute_timeout_s": args.precompute_timeout_s,
        "generation_timeout_s": args.generation_timeout_s,
        "timed_out": False,
        "timeout_phase": None,
        "attempt": None,
        "worker_gpu": None,
        "cuda_visible_device": None,
        "worker_subprocess_runtime_s": None,
        "progress_phase": None,
        "progress_metrics": None,
        "result": None,
        "log_path": None,
        "result_path": None,
        "launcher_error": f"{type(error).__name__}: {error}",
    }
    _write_json(run_dir / "jobs" / job.job_id / "status.json", status)
    return status


def _aggregate(
    run_dir: Path, jobs: Sequence[Job], statuses: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rows = [_flat_row(job, statuses.get(job.job_id)) for job in jobs]
    counts = Counter(str(row["status"]) for row in rows)
    failures = Counter(
        str(row["failure_kind"]) for row in rows if row["status"] == "failed"
    )
    summary = {
        "schema_version": RESULT_SCHEMA_VERSION,
        "jobs": len(rows),
        "succeeded": counts["success"],
        "failed": counts["failed"],
        "pending": counts["pending"],
        "failure_counts": dict(failures),
        "valid_generations": sum(row.get("valid_generation") is True for row in rows),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_csv(run_dir / "results.csv", rows)
    _write_json(run_dir / "summary.json", summary)
    return summary


def _resolve_gpus(raw: str | None) -> list[str]:
    inherited = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if raw:
        result = [value.strip() for value in raw.split(",") if value.strip()]
    elif inherited:
        result = [str(index) for index in range(len(inherited))]
    else:
        result = ["0", "1", "2", "3"]
    if not result or len(set(result)) != len(result):
        raise ValueError("GPU list must be nonempty and unique")
    return result


def _worker_cuda_device(gpu: str) -> str:
    inherited = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if not inherited:
        return gpu
    if not gpu.isdecimal() or int(gpu) >= len(inherited):
        raise ValueError(
            f"GPU selection is outside inherited CUDA visibility: {gpu}"
        )
    return inherited[int(gpu)]


def _runtime_identity(
    args: argparse.Namespace,
    gpus: Sequence[str],
) -> dict[str, Any]:
    python = Path(args.python).expanduser().resolve()
    current_python = Path(sys.executable).resolve()
    if python != current_python:
        raise ValueError(
            "worker and launcher must use the same Python environment"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the XGrammar benchmark")
    devices = []
    for gpu in gpus:
        if not gpu.isdecimal():
            raise ValueError("GPU identifiers must be visible integer indices")
        index = int(gpu)
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"GPU index is not visible: {gpu}")
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "worker_gpu": gpu,
                "cuda_visible_device": _worker_cuda_device(gpu),
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [
                    properties.major,
                    properties.minor,
                ],
            }
        )
    return {
        "python_executable": str(python),
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": {
            "torch": torch.__version__,
            "transformers": importlib.metadata.version("transformers"),
            "xgrammar": importlib.metadata.version("xgrammar"),
        },
        "cuda": {
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "pytorch_cuda_alloc_conf": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF"
            ),
        },
        "devices": devices,
        "automata_sha256": _file_digest(
            NFA_SRC / "nfa_fpras" / "automata.py"
        ),
    }


def _parse_indices(
    raw: str | None, available: int | Collection[int]
) -> set[int] | None:
    if raw is None:
        return None
    allowed = (
        set(range(available))
        if isinstance(available, int)
        else {int(value) for value in available}
    )
    result: set[int] = set()
    for part in raw.split(","):
        if "-" in part:
            low, high = (int(value) for value in part.split("-", 1))
            result.update(range(low, high + 1))
        else:
            result.add(int(part))
    if not result or not result <= allowed:
        raise ValueError(
            "selected indices are outside the execution_selection"
        )
    return result


def _pilot(workload: Workload, args: argparse.Namespace) -> dict[str, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    config = AutoConfig.from_pretrained(
        args.base_model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    width = _config_vocab_size(config)
    endpoints = [
        max(
            (job for job in workload.jobs if job.source["constraint"] == family),
            key=lambda job: (job.source["unrolled_states"], job.design_index),
        )
        for family in workload_family_order(workload)
    ]
    results = []
    for job in endpoints:
        artifact = token_nfa_from_compiled(
            tokenizer,
            width,
            job.compiled,
            args.terminal_token_ids,
        )
        ebnf, metadata = nfa_to_bounded_ebnf(
            artifact, job.compiled.n_low, job.compiled.n
        )
        compiled = xgr.GrammarCompiler(
            artifact.tokenizer_info,
            max_threads=args.cpu_threads_per_worker,
            cache_enabled=False,
        ).compile_grammar(xgr.Grammar.from_ebnf(ebnf))
        audit = audit_initial_mask(compiled, artifact, metadata)
        results.append(
            {
                "constraint": job.source["constraint"],
                "job_id": job.job_id,
                "grammar_rules": metadata.rule_count,
                "grammar_edges": metadata.edge_count,
                "mask_audit_ok": audit["ok"],
            }
        )
    return {"checked": len(results), "all_passed": all(r["mask_audit_ok"] for r in results), "results": results}


def workload_family_order(workload: Workload) -> list[str]:
    return list(dict.fromkeys(str(job.source["constraint"]) for job in workload.jobs))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", type=Path)
    parser.add_argument("--base_model_path", default=str(DEFAULT_MODEL))
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--gpus")
    parser.add_argument("--timeout_s", type=float, default=256.0)
    parser.add_argument("--precompute_timeout_s", type=float)
    parser.add_argument("--generation_timeout_s", type=float)
    parser.add_argument("--cpu_threads_per_worker", type=int, default=8)
    parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_dir", type=Path)
    parser.add_argument("--indices")
    parser.add_argument("--max_jobs", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry_failed", action="store_true")
    parser.add_argument("--aggregate_only", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--pilot_grammar", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--internal-worker", action="store_true", help=argparse.SUPPRESS)
    return parser


def _launcher(args: argparse.Namespace) -> int:
    if args.workload is None:
        raise ValueError("--workload is required")
    args.precompute_timeout_s = (
        args.timeout_s
        if args.precompute_timeout_s is None
        else args.precompute_timeout_s
    )
    args.generation_timeout_s = (
        args.timeout_s
        if args.generation_timeout_s is None
        else args.generation_timeout_s
    )
    if (
        not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0
        or not math.isfinite(args.precompute_timeout_s)
        or args.precompute_timeout_s <= 0
        or not math.isfinite(args.generation_timeout_s)
        or args.generation_timeout_s <= 0
        or args.cpu_threads_per_worker <= 0
        or not math.isfinite(args.temperature)
        or args.temperature <= 0
        or not math.isfinite(args.top_p)
        or not 0 < args.top_p <= 1
        or args.top_k < 0
        or (args.max_jobs is not None and args.max_jobs <= 0)
    ):
        raise ValueError(
            "timeouts, CPU threads, sampling settings, or max_jobs are invalid"
        )
    if args.retry_failed and not args.resume:
        raise ValueError("--retry_failed requires --resume")
    if (args.resume or args.aggregate_only) and args.run_dir is None:
        raise ValueError("--resume/--aggregate_only requires --run_dir")
    workload = Workload.load(args.workload.resolve())
    selected = _parse_indices(
        args.indices, {job.design_index for job in workload.jobs}
    )
    gpus = _resolve_gpus(args.gpus)
    if args.dry_run:
        result = {
            "workload": str(workload.path),
            "dataset": workload.dataset,
            "benchmark_name": workload.benchmark_name,
            "jobs": len(workload.jobs),
            "families": dict(Counter(job.source["constraint"] for job in workload.jobs)),
            "nfa_states_range": [
                min(job.source["nfa_states"] for job in workload.jobs),
                max(job.source["nfa_states"] for job in workload.jobs),
            ],
            "unrolled_states_range": [
                min(job.source["unrolled_states"] for job in workload.jobs),
                max(job.source["unrolled_states"] for job in workload.jobs),
            ],
            "workload_file_sha256": workload.file_sha256,
            "jobs_sha256": workload.jobs_sha256,
            "gpus": gpus,
        }
        print(json.dumps(result, indent=2))
        return 0
    if args.pilot_grammar:
        model_path = Path(args.base_model_path).expanduser().resolve()
        args.base_model_path = str(model_path)
        _validate_model_snapshot(model_path, workload.model_provenance)
        _validate_tokenizer_files(
            model_path,
            workload.jobs[0].compiled.tokenizer_fingerprint,
        )
        terminal_policy = _load_terminal_policy(
            model_path,
            trust_remote_code=args.trust_remote_code,
            local_files_only=args.local_files_only,
        )
        args.terminal_token_ids = terminal_policy["terminal_token_ids"]
        print(json.dumps(_pilot(workload, args), indent=2))
        return 0
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = (
        args.run_dir.resolve()
        if args.run_dir
        else (args.output_root / f"xgrammar_{workload.dataset}_500_{timestamp}").resolve()
    )
    manifest_path = run_dir / "manifest.json"
    model_path = Path(args.base_model_path).expanduser().resolve()
    args.base_model_path = str(model_path)
    args.python = Path(args.python).expanduser().resolve()
    runtime_identity = _runtime_identity(args, gpus)
    model_identity = _validate_model_snapshot(
        model_path,
        workload.model_provenance,
    )
    tokenizer_fingerprint_sha256 = _validate_tokenizer_files(
        model_path,
        workload.jobs[0].compiled.tokenizer_fingerprint,
    )
    terminal_policy = _load_terminal_policy(
        model_path,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
    )
    args.terminal_token_ids = terminal_policy["terminal_token_ids"]
    frozen = {
        "runner_version": RUNNER_VERSION,
        "result_schema_version": RESULT_SCHEMA_VERSION,
        "method": METHOD,
        "dataset": workload.dataset,
        "benchmark_name": workload.benchmark_name,
        "workload_kind": WORKLOAD_KIND,
        "workload_schema_version": WORKLOAD_SCHEMA_VERSION,
        "workload_sha256": workload.file_sha256,
        "jobs_sha256": workload.jobs_sha256,
        "workload_file_sha256": workload.file_sha256,
        "model_provenance": workload.model_provenance,
        "base_model_path": str(model_path),
        "base_model_identity": model_identity,
        "tokenizer_fingerprint_sha256": tokenizer_fingerprint_sha256,
        "terminal_policy": terminal_policy,
        "python": str(args.python),
        "runtime_identity": runtime_identity,
        "worker_gpus": gpus,
        "trust_remote_code": args.trust_remote_code,
        "local_files_only": args.local_files_only,
        "dtype": args.dtype,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "timeout_s": args.timeout_s,
        "precompute_timeout_s": args.precompute_timeout_s,
        "generation_timeout_s": args.generation_timeout_s,
        "cpu_threads_per_worker": args.cpu_threads_per_worker,
        "runner_sha256": _file_digest(SCRIPT),
    }
    run_contract_sha256 = _digest(frozen)
    if manifest_path.is_file():
        if not (args.resume or args.aggregate_only):
            raise FileExistsError(f"run already exists: {run_dir}")
        manifest = json.loads(manifest_path.read_text())
        changed = {
            key: (manifest.get(key), value)
            for key, value in frozen.items()
            if manifest.get(key) != value
        }
        if manifest.get("run_contract_sha256") != run_contract_sha256:
            changed["run_contract_sha256"] = (
                manifest.get("run_contract_sha256"),
                run_contract_sha256,
            )
        if changed:
            raise ValueError(f"resume configuration changed: {changed}")
    else:
        if args.resume or args.aggregate_only:
            raise FileNotFoundError(f"run manifest is missing: {manifest_path}")
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(
            manifest_path,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_contract": "embedded_compiled_constraint_only",
                "precompute_runtime_definition": (
                    "compiled NFA/token partition to bounded GBNF plus uncached "
                    "XGrammar compilation; model loading and generation excluded"
                ),
                "environment": runtime_identity,
                "run_contract_sha256": run_contract_sha256,
                **frozen,
            },
        )
        (run_dir / "workload.json").write_bytes(workload.raw)
        _write_csv(run_dir / "plan.csv", [job.identity() for job in workload.jobs])
    statuses = _load_statuses(
        run_dir,
        workload.jobs,
        run_contract_sha256,
        {
            "timeout_s": args.timeout_s,
            "precompute_timeout_s": args.precompute_timeout_s,
            "generation_timeout_s": args.generation_timeout_s,
        },
        args.terminal_token_ids,
    )
    if args.aggregate_only:
        print(json.dumps(_aggregate(run_dir, workload.jobs, statuses), indent=2))
        return 0
    candidates = [
        job
        for job in workload.jobs
        if (selected is None or job.design_index in selected)
        and (
            job.job_id not in statuses
            or (args.retry_failed and statuses[job.job_id]["status"] == "failed")
        )
    ]
    if args.max_jobs is not None:
        candidates = candidates[: args.max_jobs]
    if not candidates:
        print(json.dumps(_aggregate(run_dir, workload.jobs, statuses), indent=2))
        return 0
    gpu_pool: "queue.Queue[str]" = queue.Queue()
    for gpu in gpus:
        gpu_pool.put(gpu)
    atexit.register(_terminate_all)
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                _run_one,
                args,
                run_dir,
                job,
                gpu_pool,
                run_contract_sha256,
            ): job
            for job in candidates
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            job = futures[future]
            try:
                status = future.result()
            except Exception as error:
                status = _launcher_error_status(
                    args,
                    run_dir,
                    job,
                    run_contract_sha256,
                    error,
                )
            statuses[job.job_id] = status
            _aggregate(run_dir, workload.jobs, statuses)
            print(
                f"[{completed}/{len(candidates)}] {job.job_id}: "
                f"{status['status']}"
                + (
                    ""
                    if status["status"] == "success"
                    else f"/{status.get('failure_kind')}"
                ),
                flush=True,
            )
    print(json.dumps(_aggregate(run_dir, workload.jobs, statuses), indent=2))
    return 0


def main() -> int:
    argv = sys.argv[1:]
    if argv and argv[0] == "--internal-worker":
        return _worker_main(argv[1:])
    return _launcher(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
