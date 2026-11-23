# Mapping residency on Android

`mapping_observer.c` samples the process RSS and swap from `/proc/self/smaps`.
For the CPU weights, NPU weights, and QNN context, it also records how many
mapped KiB `mincore` reports resident. Both are snapshots. A process-level major
fault count cannot identify the file that triggered a fault.

Use the phase-fault runner from commit `2d0ca247`. It must import `getrusage`;
the observer uses that runner's phase-call order. Build on the target with
`clang -O2 -fPIC -shared mapping_observer.c -o mapping_observer.so -ldl`.
Set `MLLM_SHADOWNPU_PHASE_FAULTS=1`. Set `SHADOWNPU_MAP_RUNNER` to the
absolute runner path and pass the corresponding absolute file paths in
`SHADOWNPU_MAP_CPU`, `SHADOWNPU_MAP_NPU`, and `SHADOWNPU_MAP_CONTEXT`.
Prepend the shared object to `LD_PRELOAD` so the regular FastRPC preload
remains in the environment.

Accept a trace only when all 22 `SHADOWNPU_PROCESS_FAULTS` records appear in
the expected order and the `SHADOWNPU_MAPPING` records align with the matching
zero-based calls. Check the file mappings and require zero `mincore_errors`
before using the residency fields. Keep the observer outside performance
samples. Its overhead is recorded in `elapsed_us`.
