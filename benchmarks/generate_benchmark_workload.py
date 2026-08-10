#!/usr/bin/env python3
"""Generate one immutable JSONL workload shared by all A/B/C benchmark groups."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

try:
    from benchmark_workload import (
        TokenTextFactory,
        generate_workload,
        load_profile,
        override_system_prompt_tokens,
        workload_manifest,
        write_workload_jsonl,
    )
except ModuleNotFoundError:  # Imported as benchmarks.generate_benchmark_workload in tests.
    from benchmarks.benchmark_workload import (
        TokenTextFactory,
        generate_workload,
        load_profile,
        override_system_prompt_tokens,
        workload_manifest,
        write_workload_jsonl,
    )


class ServerTokenizer:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None,
        timeout: float,
        verify: bool,
    ):
        normalized = base_url.rstrip("/") + "/"
        self.tokenize_url = urljoin(normalized, "tokenize")
        self.detokenize_url = urljoin(normalized, "detokenize")
        self.model = model
        self.timeout = timeout
        self.verify = verify
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._local = threading.local()
        self._lock = threading.Lock()
        self._sessions: list[requests.Session] = []
        self.tokenize_calls = 0

    def _http(self) -> requests.Session:
        session = getattr(self._local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(self._headers)
            self._local.session = session
            with self._lock:
                self._sessions.append(session)
        return session

    def tokenize(self, prompt: str) -> list[int]:
        with self._lock:
            self.tokenize_calls += 1
            tokenize_calls = self.tokenize_calls
        if tokenize_calls == 1 or tokenize_calls % 25 == 0:
            print(
                f"[workload] tokenizer calls={tokenize_calls}",
                file=sys.stderr,
                flush=True,
            )
        response = self._http().post(
            self.tokenize_url,
            json={"model": self.model, "prompt": prompt, "add_special_tokens": False},
            timeout=self.timeout,
            verify=self.verify,
        )
        response.raise_for_status()
        value = response.json()
        tokens = value.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(token, int) for token in tokens):
            raise ValueError(f"invalid /tokenize response from {self.tokenize_url}")
        return tokens

    def detokenize(self, tokens: list[int]) -> str:
        response = self._http().post(
            self.detokenize_url,
            json={"model": self.model, "tokens": tokens},
            timeout=self.timeout,
            verify=self.verify,
        )
        response.raise_for_status()
        value = response.json()
        prompt = value.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError(f"invalid /detokenize response from {self.detokenize_url}")
        return prompt

    def close(self) -> None:
        for session in self._sessions:
            session.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tokenizer-url",
        help="Direct vLLM server root exposing /tokenize and /detokenize.",
    )
    parser.add_argument("--model", help="Served model name; required with --tokenizer-url.")
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the profile seed for this repetition without editing the profile.",
    )
    parser.add_argument("--tokenizer-concurrency", type=int, default=16)
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--insecure", action="store_true")
    system_prompt_env = os.environ.get("SYSTEM_PROMPT_TOKENS")
    if system_prompt_env:
        try:
            system_prompt_default = int(system_prompt_env)
        except ValueError:
            parser.error("SYSTEM_PROMPT_TOKENS must be a positive integer")
    else:
        system_prompt_default = None
    parser.add_argument(
        "--system-prompt-tokens",
        type=int,
        default=system_prompt_default,
        help="Override data.system_prompt_tokens without editing the profile (also reads SYSTEM_PROMPT_TOKENS).",
    )
    parser.add_argument(
        "--allow-unverified-token-counts",
        action="store_true",
        help="Generate token-like text without a model tokenizer. Intended only for local tests.",
    )
    args = parser.parse_args()
    if args.tokenizer_url and not args.model:
        parser.error("--model is required with --tokenizer-url")
    if not args.tokenizer_url and not args.allow_unverified_token_counts:
        parser.error(
            "--tokenizer-url is required for production workloads; "
            "use --allow-unverified-token-counts only for local smoke tests"
        )
    if args.system_prompt_tokens is not None and args.system_prompt_tokens <= 0:
        parser.error("--system-prompt-tokens must be a positive integer")
    if args.tokenizer_concurrency <= 0:
        parser.error("--tokenizer-concurrency must be a positive integer")
    return args


def main() -> int:
    args = parse_args()
    profile = override_system_prompt_tokens(load_profile(args.profile), args.system_prompt_tokens)
    if args.seed is not None:
        profile = dict(profile)
        profile["seed"] = args.seed
    started = time.monotonic()
    print(
        f"[workload] loading profile={args.profile} tokenizer={args.tokenizer_url or 'unverified-smoke'}",
        file=sys.stderr,
        flush=True,
    )
    tokenizer: ServerTokenizer | None = None
    try:
        if args.tokenizer_url:
            tokenizer = ServerTokenizer(
                args.tokenizer_url,
                args.model,
                os.environ.get(args.api_key_env),
                args.timeout,
                not args.insecure,
            )
            factory = TokenTextFactory(
                tokenizer.tokenize,
                tokenizer.detokenize,
                max_workers=args.tokenizer_concurrency,
            )
        else:
            factory = TokenTextFactory()
        records = generate_workload(profile, factory)
        print(
            f"[workload] generated {len(records)} requests in {time.monotonic() - started:.1f}s; writing JSONL",
            file=sys.stderr,
            flush=True,
        )
        write_workload_jsonl(args.output, records)
        manifest = workload_manifest(profile, records, factory.verified)
        manifest["sha256"] = hashlib.sha256(args.output.read_bytes()).hexdigest()
        manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    finally:
        if tokenizer is not None:
            tokenizer.close()

    print(
        json.dumps(
            {
                "workload": str(args.output),
                "manifest": str(manifest_path),
                "requests": len(records),
                "token_count_verified": factory.verified,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
