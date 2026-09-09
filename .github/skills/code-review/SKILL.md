---
name: code-review
description: Reviews KernelPort pull requests for reliability and scalability defects in the Rust inference server and Python sidecar workers. Use when reviewing changes to request handling, batching, scheduling, worker execution, backend integration, tensor handling, or deployment scripts. Prioritizes silent failures, unvalidated input at trust boundaries, blocking calls in async tasks, missing backpressure, and broken batch invariants over style nits.
---

# KernelPort Code Review

KernelPort is a Rust gRPC inference server that fans requests through a batcher →
scheduler → worker pipeline into native backends (ONNX Runtime, Helion sidecar).
It runs on GPUs under concurrent load. A defect here does not produce a stack
trace in someone's editor — it produces **wrong tensors returned to a client**, a
**hung worker**, or **latency collapse under load**.

Review accordingly. Optimize for finding the small number of defects that
matter, not for coverage of every observation.

## Severity ladder

Label every comment with one of these. Do not post a comment you cannot label.

- **`blocking`** — Returns incorrect results, drops or corrupts a request,
  deadlocks, panics on the request path, leaks a secret, or degrades the server
  permanently. Merging this causes a production incident.
- **`important`** — Correct today but fails under concurrency, load, or a
  reachable edge case. Or: removes an invariant that later code will rely on.
- **`optional`** — A genuine improvement in clarity or efficiency with no
  correctness impact.

**Noise budget: at most 3 `optional` comments per PR.** If you have more than
that, you are reviewing style instead of behavior. Drop the weakest ones.

If the diff is clean, say so in one line. Do not manufacture findings.

## 1. Silent failure — highest priority

This codebase has a standing tendency to paper over bad input with a default
value instead of rejecting it. Flag every instance as `blocking`.

**Lossy conversion that fabricates data:**

```rust
// ❌ A negative or oversized dimension silently becomes a 0-length axis.
//    The request proceeds and returns a wrong-shaped tensor.
let shape: Vec<usize> = t.shape.into_iter()
    .map(|d| usize::try_from(d).unwrap_or(0))
    .collect();

// ✅ Reject at the boundary.
let shape: Vec<usize> = t.shape.into_iter()
    .map(|d| usize::try_from(d)
        .map_err(|_| Status::invalid_argument(format!("invalid dimension {d}"))))
    .collect::<Result<_, _>>()?;
```

**An unhandled enum arm that returns an empty success:**

```rust
// ❌ A CUDA-resident output becomes an empty byte buffer and a 200 response.
//    The client cannot tell this from a real result.
TensorStorage::CudaDevice(_) => Bytes::new(),

// ✅ Unimplemented paths must fail loudly.
TensorStorage::CudaDevice(_) => {
    return Err(Status::unimplemented("device-resident outputs not supported"));
}
```

Also flag: `unwrap_or_default()` on parsed external input, `.ok()` that discards
an error, empty `catch`/`except` blocks, and any fallback that substitutes a
plausible-looking value for a failed operation.

## 2. Trust-boundary validation

`crates/kernelport-server/src/grpc.rs` is the only place untrusted bytes enter
the system. Every field of `InferRequest` is attacker-controlled.

Require, before a tensor is constructed:

- `data.len()` equals `shape.numel() * dtype.size_bytes()` — a mismatch means
  the backend reads a buffer of the wrong length.
- Shape rank and per-dimension bounds are checked.
- `numel()` is computed with checked arithmetic. `iter().product::<usize>()`
  wraps silently in release builds; a crafted shape can produce a tiny `numel`
  for a huge allocation.
- The requested model and dtype are known values.

```rust
// ✅ Size agreement is the invariant every downstream backend assumes.
let expected = shape.numel()
    .checked_mul(dtype.size_bytes())
    .ok_or_else(|| Status::invalid_argument("shape overflows address space"))?;
if data.len() != expected {
    return Err(Status::invalid_argument(
        format!("expected {expected} bytes for shape/dtype, got {}", data.len())));
}
```

Flag any new `Tensor::from_cpu_bytes` call reachable from the network that does
not validate first.

## 3. Pipeline invariants: batcher, scheduler, worker

These three loops are the reliability core. Hold them to the strictest standard.

**Batch merge must actually merge.** If N requests are batched, all N responses
must be derived from their own inputs. Reusing one request's inputs for the
batch, or broadcasting one output to every waiting `resp_tx`, returns **wrong
inference results to real clients** — `blocking`, always, even if a comment says
`v0` or `TODO`.

**A supervisor loop must not die on a per-item error.** `run()` returning `Err`
takes the stage down for the whole process lifetime; the stage upstream then
blocks or errors forever.

```rust
// ❌ One bad inference kills this worker permanently. The scheduler keeps
//    round-robining into a dead channel.
while let Some(job) = self.inbox.recv().await {
    self.model.infer_batch(job)?;
}

// ✅ Fail the job, keep the loop alive, make the failure observable.
while let Some(job) = self.inbox.recv().await {
    if let Err(e) = self.model.infer_batch(job) {
        error!(worker_id = self.id, error = %e, "batch failed");
        // and: send Err(..) to every resp_tx in the batch
    }
}
```

**Every request owns a `oneshot::Sender` that must be resolved.** If a job is
dropped on an error path, its senders drop with it and callers see a generic
"worker dropped" internal error instead of the real cause. On any early return
that discards a `BatchJob`, require that each request's `resp_tx` receives a
descriptive `Err` first.

**Deadlines must be enforced, not just stored.** `InferenceRequest::deadline`
exists; check that queued work is dropped once it is past due rather than run
against a client that has already given up. Flag hardcoded deadlines that ignore
the client's gRPC deadline.

**Scheduler distribution.** Flag `% worker_txs.len()` without a non-empty
guard — that is a divide-by-zero panic. Prefer distribution by queue depth or
readiness over blind round-robin; one slow worker head-of-line blocks its share
of traffic.

## 4. Async and blocking — scale defects

The server is single-runtime Tokio. A blocking call inside an async task stalls
an executor thread and, under load, the whole runtime.

```rust
// ❌ Synchronous backend inference (ORT / FFI / GPU sync) inside an async task.
self.model.infer_batch(job)?;

// ✅ Move CPU/GPU-bound work off the async runtime.
let result = tokio::task::spawn_blocking(move || model.infer_batch(job)).await?;
```

Flag as `blocking` or `important`:

- Synchronous FFI, ORT calls, CUDA sync, `std::fs`, `std::thread::sleep`, or
  long compute inside `async fn` or a `tokio::select!` arm.
- A `std::sync::Mutex` guard held across an `.await`.
- `block_on` inside async context.

**Backpressure and load shedding.** An `mpsc::Sender::send().await` on a full
channel waits without bound — latency grows silently instead of the server
rejecting work. At the gRPC boundary prefer `try_send` and map a full queue to
`Status::resource_exhausted`, so callers can retry or shed. Flag any new
unbounded channel or any unbounded `Vec` accumulator on the request path.

**Copies on the hot path.** Tensor payloads are large. Flag `.to_vec()`,
`.clone()`, or `Bytes` → `Vec<u8>` conversions per request where the buffer
could be moved or reference-counted. Note the size class in the comment — a
copy of a 4-element test tensor is not worth flagging.

## 5. Rust specifics

- **No panics on the request path.** `unwrap()`, `expect()`, `panic!`,
  slice indexing, and integer division are `blocking` in `grpc.rs`, the runtime
  crates, and backend code. They are acceptable in `tests/`, `build.rs`, and
  startup-time configuration that should fail fast.
- **Arithmetic on external values** uses `checked_*` / `saturating_*`.
- **`unsafe`** requires a `// SAFETY:` comment naming the invariant the caller
  must uphold. Missing justification is `blocking`.
- **Errors crossing the gRPC boundary** must map to a correct `Status` code:
  bad input → `invalid_argument`, queue full → `resource_exhausted`, unknown
  model → `not_found`, missing feature → `unimplemented`. Do not collapse
  everything into `internal`, and do not leak internal paths or config values
  in the message.
- Do not comment on formatting or lint output. `cargo fmt --all -- --check` and
  `cargo clippy --all-targets --all-features -- -D warnings` are enforced in CI
  and will catch it.

## 6. Python sidecars and scripts

Applies to `scripts/helion/`, `scripts/luxtts/`, `scripts/lambda/`,
`scripts/validation/`.

- **Worker handlers must not crash the server.** An exception in a gRPC handler
  should return a status to the caller and keep the process serving.
- **Validate tensor payloads on arrival**, same rule as §2: dtype, shape, and
  byte length must agree before reaching `numpy`/`torch`.
- **Bound resource growth** — no per-request accumulation in module-level state,
  no unbounded caches.
- **Subprocess calls** use argument lists, never `shell=True` with interpolated
  values. Flag string-built shell commands in deploy scripts as `blocking`.
- **No secrets in logs or command lines.** `HF_TOKEN`,
  `HUGGINGFACE_HUB_TOKEN`, `LAMBDA_CLOUD_API_KEY`, and `LAMBDA_SSH_PRIVATE_KEY`
  must not be echoed, printed, interpolated into a remote command, or written
  to a results artifact.
- **Cloud resources must be released on every exit path**, including exceptions
  and cancellation. A leaked Lambda GPU instance bills continuously — flag any
  provisioning without teardown in a `finally` or trap.

## 7. Tests

Judge whether the test would actually catch the regression the change could
introduce.

- New backend, dtype, or shape handling needs a case covering the **invalid**
  input, not only the happy path.
- Changes to batching, scheduling, or worker lifecycle need a **concurrent**
  test with **distinct inputs per request** — a batch test where every request
  sends identical tensors cannot detect cross-request contamination. Flag this
  specifically; it is the most likely gap in this repo.
- Flag assertions that only check "no error" for logic that produces values.
- Do not require tests for docs, comments, or pure formatting changes.

## What not to comment on

Staying silent on these is what makes the rest of the review worth reading.

- Formatting, import order, line length — `cargo fmt` and `clippy` own these.
- Naming preferences, unless the name is actively misleading about behavior.
- Rewriting working code in a different but equivalent style.
- Missing docs on private helpers.
- Speculative future requirements the PR does not claim to address.
- `v0` / `TODO` scaffolding that is explicitly scoped as such **and** cannot
  return a wrong result to a client. If it can return a wrong result, flag it —
  the `v0` label does not make silent corruption acceptable.
- Repeating a finding you already made elsewhere in the same PR. Comment once at
  the clearest instance and note that it recurs.

## Comment format

Each comment: the severity label, one sentence naming the concrete failure
(input or condition → observable result), then the fix as a diff or code block.

> **`blocking`** — A client sending `shape: [-1, 8]` gets a tensor with a
> 0-length axis and a success response instead of `INVALID_ARGUMENT`, because
> `try_from(d).unwrap_or(0)` swallows the conversion error.
>
> ```rust
> .map(|d| usize::try_from(d)
>     .map_err(|_| Status::invalid_argument(format!("invalid dimension {d}"))))
> .collect::<Result<Vec<_>, _>>()?
> ```

No preamble, no summary of what the PR does, no praise. If you are unsure a
finding is real, say what you would need to confirm it rather than asserting it.
