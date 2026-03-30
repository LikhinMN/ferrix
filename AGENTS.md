# AGENTS.md

## Scope and source of truth
- This file applies to the whole repository.
- A glob search for common AI instruction files (`.github/copilot-instructions.md`, `AGENT.md`, `CLAUDE.md`, `.cursorrules`, `README.md`, etc.) found no matches, so rely on crate files directly.

## Project map (Rust library crate)
- Crate manifest: `Cargo.toml`.
- Main implementation: `src/lib.rs`.
- Unit tests: `src/tests.rs`, included from `src/lib.rs` via `#[cfg(test)] mod tests;`.
- Build outputs are under `target/` and should not be edited.

## Architecture and data model
- Core type is `NDArray<T>` in `src/lib.rs` with three coupled fields:
  - `data: Vec<T>` (flat storage)
  - `shape: Vec<usize>` (dimensions)
  - `strides: Vec<usize>` (row-major index mapping)
- `NDArray::new(data, shape)` is the constructor and invariant gate:
  - Validates `data.len() == shape.iter().product()`.
  - Computes row-major strides from right to left (`[2,3,4] -> [12,4,1]`).
- `NDArray::get(&self, index: &[usize])` exists as a stub (currently empty), so any indexing feature should be implemented there first.

## Dependency and integration points
- Library types: `crate-type = ["cdylib", "rlib"]` in `Cargo.toml`.
  - `rlib` is used by Rust unit tests.
  - `cdylib` indicates planned FFI/dynamic library integration; avoid Rust-only assumptions if adding public APIs.
- Dependencies present: `rayon`, `num-traits` (not yet used in current source).
- Dev dependency: `criterion` (benchmarking scaffolding present in manifest, no benchmark files yet).

## Tested behavior and current edge cases
- Stride behavior is validated in `src/tests.rs`:
  - `vec![2, 3]` expects strides `vec![3, 1]`.
  - `vec![2, 3, 4]` expects strides `vec![12, 4, 1]`.
- Panic-path test currently fails because expected panic text does not match `NDArray::new` message. Keep panic assertions and panic strings synchronized when editing validation logic.

## Developer workflow
- Run tests: `cargo test`.
- Re-run library tests only: `cargo test --lib`.
- Show backtraces for panic debugging: `RUST_BACKTRACE=1 cargo test --lib`.
- Current baseline (as of 2026-03-30): `cargo test` fails in `tests::test_invalid` due to panic message mismatch, and compiler warns about unused `get`/`index`.

## Code change conventions seen in this repo
- Validation errors in constructors currently use `panic!` with detailed context (`data len`, `shape`, `expected`).
- Tests are colocated as `src/tests.rs` (not under `tests/` integration-test directory).
- Generic container design (`NDArray<T>`) is already in place; preserve generic signatures unless there is a strong reason to specialize.

