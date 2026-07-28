"""Exact measured-kernel lookup for DeepSeek-V4."""

from __future__ import annotations

import csv
from pathlib import Path


class MissingDSV4BenchmarkData(RuntimeError):
    """Raised when a V4 simulation has no exact measured row."""


def _normalize(value):
    return str(value).strip().lower()


def canonical_dsv4_attention_backend(value):
    value = _normalize(value)
    return {
        "compressed": "dsv4",
        "sglang_hopper": "dsv4",
    }.get(value, value)


def canonical_dsv4_kv_dtype(value):
    value = _normalize(value)
    aliases = {
        "fp8": "fp8_e4m3",
        "e4m3": "fp8_e4m3",
        "fp8_e4m3": "fp8_e4m3",
        "fp8_e4m3fn": "fp8_e4m3",
        "float8_e4m3fn": "fp8_e4m3",
        "bf16": "bfloat16",
        "bfloat16": "bfloat16",
    }
    try:
        return aliases[value]
    except KeyError as error:
        raise ValueError(
            "DeepSeek-V4 KV dtype must be fp8_e4m3 or bfloat16; " f"got {value!r}"
        ) from error


def canonical_dsv4_moe_key(backend, expert_weight_dtype):
    backend = _normalize(backend)
    backend = {
        "marlin_w4a16": "marlin",
        "w4a16_marlin": "marlin",
    }.get(backend, backend)
    expert_weight_dtype = _normalize(expert_weight_dtype)
    if backend == "auto":
        raise ValueError(
            "DeepSeek-V4 benchmark rows require the resolved MoE backend, " "not 'auto'"
        )
    if backend == "flashinfer_mxfp4" and expert_weight_dtype != "fp4":
        raise ValueError("flashinfer_mxfp4 benchmark rows require fp4 expert weights")
    return backend, expert_weight_dtype


class DSV4BenchmarkData:
    """Read exact V4 layer-component latencies from ``bench_data/dsv4``."""

    _REQUIRED_COLUMNS = {
        "attention.csv": {
            "architecture_fingerprint",
            "tp_size",
            "backend",
            "attention_weight_dtype",
            "kv_dtype",
            "compress_state_dtype",
            "ratio",
            "batch_size",
            "q_len",
            "past_len",
            "latency_us",
        },
        "mhc.csv": {
            "architecture_fingerprint",
            "tp_size",
            "backend",
            "attention_weight_dtype",
            "batch_size",
            "q_len",
            "mode",
            "latency_us",
        },
        "moe.csv": {
            "architecture_fingerprint",
            "tp_size",
            "world_size",
            "num_nodes",
            "backend",
            "expert_weight_dtype",
            "batch_size",
            "q_len",
            "latency_us",
        },
    }

    def __init__(self, root=None, architecture_fingerprint=None):
        if root is None:
            root = Path(__file__).resolve().parents[1] / "bench_data" / "dsv4"
        self.root = Path(root)
        self.architecture_fingerprint = architecture_fingerprint
        self._cache = {}

    def _path(self, phase, device_type, file_name):
        return self.root / phase / device_type.lower() / file_name

    @staticmethod
    def _canonicalize(file_name, values):
        values = dict(values)
        if file_name in {"attention.csv", "mhc.csv"}:
            values["backend"] = canonical_dsv4_attention_backend(values["backend"])
        if file_name == "attention.csv":
            values["kv_dtype"] = canonical_dsv4_kv_dtype(values["kv_dtype"])
        if file_name == "moe.csv":
            (
                values["backend"],
                values["expert_weight_dtype"],
            ) = canonical_dsv4_moe_key(values["backend"], values["expert_weight_dtype"])
        return values

    def _read_rows(self, phase, device_type, file_name):
        path = self._path(phase, device_type, file_name)
        if path in self._cache:
            return self._cache[path]
        if not path.exists():
            raise MissingDSV4BenchmarkData(f"Missing benchmark file: {path}")

        with path.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            fields = set(reader.fieldnames or [])
            missing = self._REQUIRED_COLUMNS[file_name] - fields
            if missing:
                raise MissingDSV4BenchmarkData(
                    f"{path} is missing columns: {', '.join(sorted(missing))}"
                )
            rows = [
                self._canonicalize(file_name, row)
                for row in reader
                if any(row.values())
            ]
        self._cache[path] = rows
        return rows

    def _exact_latency(self, phase, device_type, file_name, filters):
        path = self._path(phase, device_type, file_name)
        filters = self._canonicalize(file_name, filters)
        if self.architecture_fingerprint is not None:
            filters["architecture_fingerprint"] = self.architecture_fingerprint

        matches = [
            row
            for row in self._read_rows(phase, device_type, file_name)
            if all(
                str(row.get(key, "")) == str(value) for key, value in filters.items()
            )
        ]
        description = ", ".join(f"{key}={value}" for key, value in filters.items())
        if len(matches) != 1:
            raise MissingDSV4BenchmarkData(
                f"Expected one exact row in {path} for {description}; "
                f"found {len(matches)}. DeepSeek-V4 does not use a "
                "nearest-shape or analytical fallback."
            )
        try:
            latency_us = float(matches[0]["latency_us"])
        except (TypeError, ValueError) as error:
            raise MissingDSV4BenchmarkData(
                f"Invalid latency_us in {path}: {matches[0].get('latency_us')!r}"
            ) from error
        if latency_us <= 0:
            raise MissingDSV4BenchmarkData(
                f"latency_us in {path} must be positive, got {latency_us}"
            )
        return latency_us

    def attention_latency_us(
        self,
        *,
        phase,
        device_type,
        tp_size,
        backend,
        attention_weight_dtype,
        kv_dtype,
        compress_state_dtype,
        ratio,
        batch_size,
        q_len,
        past_len,
    ):
        return self._exact_latency(
            phase,
            device_type,
            "attention.csv",
            {
                "tp_size": tp_size,
                "backend": backend,
                "attention_weight_dtype": attention_weight_dtype,
                "kv_dtype": kv_dtype,
                "compress_state_dtype": compress_state_dtype,
                "ratio": ratio,
                "batch_size": batch_size,
                "q_len": q_len,
                "past_len": past_len,
            },
        )

    def mhc_latency_us(
        self,
        *,
        phase,
        device_type,
        tp_size,
        backend,
        attention_weight_dtype,
        batch_size,
        q_len,
        mode,
    ):
        return self._exact_latency(
            phase,
            device_type,
            "mhc.csv",
            {
                "tp_size": tp_size,
                "backend": backend,
                "attention_weight_dtype": attention_weight_dtype,
                "batch_size": batch_size,
                "q_len": q_len,
                "mode": mode,
            },
        )

    def moe_latency_us(
        self,
        *,
        phase,
        device_type,
        tp_size,
        world_size,
        num_nodes,
        backend,
        expert_weight_dtype,
        batch_size,
        q_len,
    ):
        return self._exact_latency(
            phase,
            device_type,
            "moe.csv",
            {
                "tp_size": tp_size,
                "world_size": world_size,
                "num_nodes": num_nodes,
                "backend": backend,
                "expert_weight_dtype": expert_weight_dtype,
                "batch_size": batch_size,
                "q_len": q_len,
            },
        )
