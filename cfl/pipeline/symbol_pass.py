from __future__ import annotations

import ast
import json
import signal
import textwrap
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace

from cfl.core import db
from cfl.core.budget import PromptPart, assemble, batch_by_budget, split_oversized
from cfl.core.errors import BudgetExceeded, CflError, LLMValidationError
from cfl.core.hashing import ctx_hash
from cfl.core.memory import KnowledgeMemory
from cfl.parser import graph
from cfl.parser.python_adapter import PythonAdapter
from cfl.prompts import prompts, schemas


@dataclass
class PassResult:
    saved: int = 0
    cached: int = 0
    trivial: int = 0
    failed: int = 0
    llm_calls: int = 0
    interrupted: bool = False
    tokens: int = 0
    eval_ns: int = 0


def is_test(symbol):
    return any(
        p in {"test", "tests"} or p.startswith("test_") for p in symbol["file_path"].split("/")
    )


def _body(symbol):
    try:
        node = ast.parse(textwrap.dedent(symbol["raw_code"])).body[0]
        body = getattr(node, "body", [])
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        return node, body
    except (SyntaxError, IndexError):
        return None, []


def is_trivial(symbol, min_lines=0, is_test=False, skip_tests=False):
    if symbol["kind"] == "module" or (skip_tests and is_test):
        return True
    node, body = _body(symbol)
    if symbol["end_line"] - symbol["start_line"] + 1 < min_lines:
        return True
    if not body or all(isinstance(n, ast.Pass) for n in body):
        return True
    if symbol["name"] in {"__repr__", "__str__"}:
        return True
    if len(body) != 1:
        return False
    statement = body[0]
    if isinstance(statement, ast.Raise):
        value = statement.exc.func if isinstance(statement.exc, ast.Call) else statement.exc
        return isinstance(value, ast.Name) and value.id == "NotImplementedError"
    if isinstance(statement, ast.Assign):
        return (
            len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Attribute)
            and isinstance(statement.targets[0].value, ast.Name)
            and statement.targets[0].value.id == "self"
        )
    if isinstance(statement, ast.Return):
        value = statement.value
        if (
            isinstance(value, ast.Attribute)
            and isinstance(value.value, ast.Name)
            and value.value.id == "self"
        ):
            return True
        if isinstance(value, ast.Call) and isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            params = [
                p.arg
                for p in node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                if p.arg != "self"
            ]
            args = [a.id for a in value.args if isinstance(a, ast.Name)]
            args += [k.value.id for k in value.keywords if isinstance(k.value, ast.Name)]
            # Nested computations, constants and missing/extra args are not forwarding wrappers.
            return len(args) == len(value.args) + len(value.keywords) and args == params
    return False


def templated_one_liner(symbol):
    _, body = _body(symbol)
    if (
        len(body) == 1
        and isinstance(body[0], ast.Return)
        and isinstance(body[0].value, ast.Attribute)
        and isinstance(body[0].value.value, ast.Name)
        and body[0].value.value.id == "self"
    ):
        return f"Returns `{body[0].value.attr}` of the instance."
    if (
        len(body) == 1
        and isinstance(body[0], ast.Assign)
        and isinstance(body[0].targets[0], ast.Attribute)
    ):
        return f"Sets `{body[0].targets[0].attr}` of the instance."
    if symbol.get("docstring"):
        return " ".join(symbol["docstring"].split()[:25])
    if not body or all(isinstance(n, ast.Pass) for n in body):
        return f"Declares `{symbol['qualname']}` without an implementation."
    if len(body) == 1 and isinstance(body[0], ast.Return) and isinstance(body[0].value, ast.Call):
        return f"Delegates `{symbol['qualname']}` to `{ast.unparse(body[0].value.func)}`."
    return f"Defines `{symbol.get('signature') or symbol['qualname']}`."


def trivial_summary(symbol):
    line = templated_one_liner(symbol)
    return schemas.SymbolSummary(
        one_liner=line,
        purpose=symbol.get("docstring") or line,
        inputs=symbol.get("signature") or "None shown",
        returns="See source",
        side_effects=[],
        raises=[],
        notable_logic="Templated from source; no model inference.",
    )


def dependencies(conn, symbol, group, threshold=0.6, *, context=None):
    rows = context["symbols"] if context else {s["id"]: s for s in db.ctx_inputs(conn)}
    edges = context["edges"] if context else db.fetch_edges(conn, threshold)
    ids = {
        e["callee_id"] for e in edges if e["caller_id"] == symbol["id"] and e["callee_id"] in rows
    }
    if symbol["kind"] == "class":
        ids.update(
            s["id"]
            for s in rows.values()
            if s["parent_id"] == symbol["id"] and s["kind"] == "method"
        )
    pairs = [
        (id, rows[id]["summary_short"] or "No verified summary available") for id in sorted(ids)
    ]
    if len(group) > 1:
        pairs += [("@scc:" + id, rows[id]["code_hash"]) for id in sorted(group)]
    return pairs


def expected_key(conn, symbol, group, memory, *, min_lines=0, skip_tests=False, context=None):
    options = {
        **memory.options,
        "min_lines": min_lines,
        "skip_tests": skip_tests,
        "edge_conf_threshold": memory.settings.edge_conf_threshold,
    }
    return ctx_hash(
        symbol["code_hash"],
        dependencies(conn, symbol, group, memory.settings.edge_conf_threshold, context=context),
        prompts.PROMPT_VERSION,
        memory.settings.gen_model,
        gen_model_digest=memory.digests[memory.settings.gen_model],
        generation_options=options,
        schema_version=schemas.SCHEMA_VERSION,
    )


@contextmanager
def _finish_saves():
    """Defer SIGINT across the short save sequence so a completed SCC is durable."""
    pending = []
    main = threading.current_thread() is threading.main_thread()
    previous = signal.getsignal(signal.SIGINT) if main else None
    if main:
        signal.signal(signal.SIGINT, lambda *_: pending.append(True))
    try:
        yield
    finally:
        if main:
            signal.signal(signal.SIGINT, previous)
    if pending:
        raise KeyboardInterrupt


class Worker:
    def __init__(self, conn, client, settings, memory, result):
        self.conn, self.client, self.settings, self.memory, self.result = (
            conn,
            client,
            settings,
            memory,
            result,
        )
        self.context = {
            "symbols": {s["id"]: s for s in db.ctx_inputs(conn)},
            "edges": db.fetch_edges(conn, settings.edge_conf_threshold),
        }
        self.active = []
        self.raw = ""

    def attempt(self, task, symbol_id):
        if task.startswith("symbol"):
            db.record_symbol_attempt(self.conn, self.active)
            self.result.llm_calls += 1

    def structured(self, prompt, system, fmt, validate, reserve=None):
        reserve = reserve or self.settings.num_predict_symbol
        original = prompt
        for repair in range(2):
            assembled = assemble(
                system,
                [PromptPart("source", original, 1, False)]
                + (
                    [
                        PromptPart(
                            "repair",
                            "Previous output:\n"
                            + self.raw
                            + "\nValidation error: "
                            + self.error
                            + "\nReturn only valid JSON.",
                            2,
                            True,
                        )
                    ]
                    if repair
                    else []
                ),
                num_ctx=self.settings.num_ctx,
                num_predict=reserve,
                overhead=self.settings.template_overhead,
                counter=self.client.counter,
            )
            result = self.client.generate(
                assembled.prompt,
                "",
                fmt=fmt,
                num_predict=reserve,
                task="symbol_repair" if repair else "symbol",
                symbol_id=self.active[0],
            )
            self.raw = result.text
            self.result.tokens += result.output_tokens
            self.result.eval_ns += result.eval_duration
            try:
                return validate(result.text)
            except (ValueError, TypeError, KeyError) as exc:
                self.error = str(exc)
        raise LLMValidationError("Invalid output after one repair: " + self.error)

    def single(self, symbol, group):
        callees = dependencies(
            self.conn, symbol, group, self.settings.edge_conf_threshold, context=self.context
        )
        if len(group) > 1:
            peers = db.get_symbols(self.conn, [id for id in group if id != symbol["id"]])
            callees = [
                (id, line) for id, line in callees if id not in group and not id.startswith("@scc:")
            ]
            callees += [
                (p["id"], "Signature only: " + p["qualname"] + "(" + p["signature"] + ")")
                for p in peers
            ]
        source = symbol.copy()
        if symbol["kind"] == "class":
            methods = [
                s
                for s in self.context["symbols"].values()
                if s["parent_id"] == symbol["id"] and s["kind"] == "method"
            ]
            source["raw_code"] = (
                "Class "
                + symbol["qualname"]
                + "\nDocstring: "
                + (symbol.get("docstring") or "")
                + "\nAttributes: "
                + json.dumps(symbol.get("extra", {}).get("init_attrs", {}))
                + "\nMethods:\n"
                + "\n".join(
                    m["qualname"]
                    + "("
                    + m["signature"]
                    + "): "
                    + (m["summary_short"] or "No summary")
                    for m in methods
                )
            )
        system, prompt = prompts.build_symbol_prompt(source, callees)
        try:
            assemble(
                system,
                [PromptPart("target", prompt, 1, False)],
                num_ctx=self.settings.num_ctx,
                num_predict=self.settings.num_predict_symbol,
                overhead=self.settings.template_overhead,
                counter=self.client.counter,
            )
        except BudgetExceeded:
            # Keep all target statements; optional neighbor context can be dropped first.
            system, empty = prompts.build_symbol_prompt({**source, "raw_code": ""}, [])
            available = (
                self.settings.num_ctx
                - self.settings.template_overhead
                - self.settings.num_predict_symbol
                - self.client.counter.count(system + "\n\n" + empty)
                - 32
            )
            if symbol["kind"] == "class":
                lines = source["raw_code"].splitlines()
                batches = batch_by_budget(
                    lines, lambda line: self.client.counter.count(line) + 2, available
                )
                chunks = [SimpleNamespace(text="\n".join(batch)) for batch in batches]
            else:
                chunks = split_oversized(
                    source, PythonAdapter(), available, counter=self.client.counter
                )
            partials = []
            for chunk in chunks:
                _, p = prompts.build_symbol_prompt({**source, "raw_code": chunk.text}, [])
                partials.append(
                    self.structured(
                        p,
                        system,
                        schemas.SYMBOL_SUMMARY_JSON_SCHEMA,
                        schemas.SymbolSummary.model_validate_json,
                    ).model_dump_json()
                )
            # Batch reduction, never truncate or silently discard partials.
            prefix = (
                "Combine all partials into one symbol summary. Signature: "
                + symbol["signature"]
                + "\n"
            )
            budget = (
                self.settings.num_ctx
                - self.settings.template_overhead
                - self.settings.num_predict_symbol
                - self.client.counter.count(system + "\n\n" + prefix)
                - 32
            )
            while len(partials) > 1:
                batches = batch_by_budget(
                    partials, lambda p: self.client.counter.count(p) + 8, budget
                )
                if all(len(b) == 1 for b in batches):
                    raise BudgetExceeded("Partials cannot be combined within context")
                partials = [
                    self.structured(
                        prefix + "\n".join(b),
                        system,
                        schemas.SYMBOL_SUMMARY_JSON_SCHEMA,
                        schemas.SymbolSummary.model_validate_json,
                    ).model_dump_json()
                    for b in batches
                ]
            return schemas.SymbolSummary.model_validate_json(partials[0])
        return self.structured(
            prompt,
            system,
            schemas.SYMBOL_SUMMARY_JSON_SCHEMA,
            schemas.SymbolSummary.model_validate_json,
        )

    def combined(self, symbols):
        schema = {
            "type": "object",
            "properties": {s["id"]: schemas.SYMBOL_SUMMARY_JSON_SCHEMA for s in symbols},
            "required": [s["id"] for s in symbols],
            "additionalProperties": False,
        }
        group = [s["id"] for s in symbols]
        evidence = []
        for s in symbols:
            _, p = prompts.build_symbol_prompt(
                s,
                [
                    (id, line)
                    for id, line in dependencies(
                        self.conn, s, group, self.settings.edge_conf_threshold
                    )
                    if id not in group and not id.startswith("@scc:")
                ],
            )
            evidence.append("[MEMBER " + s["id"] + "]\n" + p)
        prompt = (
            "Summarize this recursion group. Return a JSON object keyed by the exact member IDs, one SymbolSummary per member.\n"
            + "\n".join(evidence)
        )
        reserve = self.settings.num_predict_symbol * len(symbols)
        assemble(
            prompts.SYSTEM_INGEST,
            [PromptPart("SCC", prompt, 1, False)],
            num_ctx=self.settings.num_ctx,
            num_predict=reserve,
            overhead=self.settings.template_overhead,
            counter=self.client.counter,
        )

        def validate(text):
            values = json.loads(text)
            if not isinstance(values, dict) or set(values) != set(group):
                raise ValueError("Missing or extra SCC members")
            return {id: schemas.SymbolSummary.model_validate(values[id]) for id in group}

        return self.structured(prompt, prompts.SYSTEM_INGEST, schema, validate, reserve)


def run_symbol_pass(
    conn,
    client,
    settings,
    *,
    resume=True,
    priority=None,
    skip_tests=False,
    min_lines=0,
    retry_failed=False,
    progress=None,
):
    if min_lines < 0:
        raise ValueError("min_lines must be nonnegative")
    memory = KnowledgeMemory(conn, client, settings)
    result = PassResult()
    worker = Worker(conn, client, settings, memory, result)
    previous = client.on_generation_attempt
    client.on_generation_attempt = worker.attempt
    started = time.monotonic()
    quarantined = 0
    try:
        items = graph.processing_order(
            conn, priority=priority, min_conf=settings.edge_conf_threshold
        )
        total = sum(len(item.symbol_ids) for item in items)
        for item in items:
            group = item.symbol_ids
            symbols = [worker.context["symbols"][id] for id in group]
            todo = []
            for s in symbols:
                key = expected_key(
                    conn,
                    s,
                    group,
                    memory,
                    min_lines=min_lines,
                    skip_tests=skip_tests,
                    context=worker.context,
                )
                valid = s["status"] in {"done", "skipped_trivial"}
                if valid:
                    try:
                        schemas.SymbolSummary.model_validate(s["summary_json"])
                    except (ValueError, TypeError):
                        valid = False
                if valid and s["ctx_hash"] == key:
                    result.cached += 1
                elif s["status"] == "failed" and not retry_failed:
                    result.failed += 1
                else:
                    todo.append(s)
            if not todo:
                if progress:
                    progress(result, total)
                continue
            generated = {}
            worker.raw = ""
            try:
                nontrivial = [
                    s for s in todo if not is_trivial(s, min_lines, is_test(s), skip_tests)
                ]
                generated.update({s["id"]: trivial_summary(s) for s in todo if s not in nontrivial})
                if (
                    len(nontrivial) > 1
                    and len(group) > 1
                    and not any(s["kind"] == "class" for s in nontrivial)
                ):
                    worker.active = [s["id"] for s in nontrivial]
                    try:
                        generated.update(worker.combined(nontrivial))
                    except BudgetExceeded:
                        for s in nontrivial:
                            worker.active = [s["id"]]
                            generated[s["id"]] = worker.single(s, group)
                else:
                    for s in sorted(nontrivial, key=lambda s: s["kind"] == "class"):
                        worker.active = [s["id"]]
                        generated[s["id"]] = worker.single(s, group)
                        worker.context["symbols"][s["id"]] = {
                            **s,
                            "summary_short": generated[s["id"]].one_liner,
                        }
                # Keys use the complete SCC output snapshot, including not-yet-saved peers.
                final_context = {
                    "symbols": dict(worker.context["symbols"]),
                    "edges": worker.context["edges"],
                }
                for s in todo:
                    summary = generated[s["id"]]
                    final_context["symbols"][s["id"]] = {
                        **s,
                        "summary_short": summary.one_liner,
                        "summary_json": summary.model_dump(),
                    }
                with _finish_saves():
                    for s in todo:
                        summary = generated[s["id"]]
                        key = expected_key(
                            conn,
                            s,
                            group,
                            memory,
                            min_lines=min_lines,
                            skip_tests=skip_tests,
                            context=final_context,
                        )
                        with conn.transaction():
                            current = db.lock_symbol_hashes(conn, group)
                            if any(
                                current.get(peer["id"]) != peer["code_hash"] for peer in symbols
                            ):
                                raise LLMValidationError("Source changed during symbol pass")
                            if is_trivial(s, min_lines, is_test(s), skip_tests):
                                db.save_trivial_summary(
                                    conn, s["id"], summary.one_liner, key, summary.model_dump()
                                )
                                result.trivial += 1
                                status = "skipped_trivial"
                            else:
                                db.save_symbol_summary(
                                    conn,
                                    s["id"],
                                    summary.model_dump(),
                                    summary.one_liner,
                                    None,
                                    key,
                                    increment_attempts=False,
                                )
                                status = "done"
                        worker.context["symbols"][s["id"]] = {
                            **final_context["symbols"][s["id"]],
                            "ctx_hash": key,
                            "status": status,
                        }
                        result.saved += 1
                rate = (result.saved - result.trivial) / max(time.monotonic() - started, 0.001)
                db.set_meta(conn, "last_rate", str(rate))
                if progress:
                    progress(result, total)
            except (CflError, ValueError, TypeError) as exc:
                for s in todo:
                    with conn.transaction():
                        db.quarantine_symbol(
                            conn, s["id"], worker.raw, str(exc), increment_attempts=False
                        )
                    worker.context["symbols"][s["id"]] = {
                        **s,
                        "summary_short": None,
                        "summary_json": None,
                        "status": "failed",
                    }
                    result.failed += 1
                    quarantined += 1
                if progress:
                    progress(result, total)
    except KeyboardInterrupt:
        result.interrupted = True
    finally:
        client.on_generation_attempt = previous
        with conn.transaction():
            if result.saved or quarantined:
                db.bump_epoch(conn)
                db.set_view_status(conn, "lexical", "stale", {})
                db.set_view_status(conn, "dense", "stale", {})
    return result


def symbol_status(conn, client, settings, *, skip_tests=False, min_lines=0):
    memory = KnowledgeMemory(conn, client, settings, persist_metadata=False)
    context = {
        "symbols": {s["id"]: s for s in db.ctx_inputs(conn)},
        "edges": db.fetch_edges(conn, settings.edge_conf_threshold),
    }
    stale = []
    for item in graph.processing_order(conn, min_conf=settings.edge_conf_threshold):
        for s in db.get_symbols(conn, item.symbol_ids):
            if s["ctx_hash"] != expected_key(
                conn,
                s,
                item.symbol_ids,
                memory,
                min_lines=min_lines,
                skip_tests=skip_tests,
                context=context,
            ):
                stale.append(s)
    rate = float(db.get_meta(conn, "last_rate") or 0)
    return {
        "counts": {
            **dict.fromkeys(("pending", "done", "failed", "skipped_trivial"), 0),
            **db.status_counts(conn),
        },
        "stale": len(stale),
        "eta_s": len(stale) / rate if rate > 0 else None,
        "failed": [
            {"id": s["id"], "error": (s["error"] or "")[:400]}
            for s in db.get_all_symbols(conn)
            if s["status"] == "failed"
        ],
        "views": db.get_view_status(conn),
    }
