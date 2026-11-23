# Conformance: interp on the ONNX v1.22.0 node tests

Compared at atol 1e-7, rtol 1e-3.

| Status | Tests |
|---|---:|
| pass | 1 |
| fail | 2 |
| unsupported | 4 |

## By op

| Op | Pass | Fail | Unsupported |
|---|---:|---:|---:|
| Add | 1 | 1 | 1 |
| Loop | 0 | 0 | 1 |
| TopK | 0 | 0 | 2 |
| composite | 0 | 1 | 0 |

## Failures

- `test_add_bcast`: data set 0, output 1: 1 of 4 elements differ
- `test_layer_norm_expanded`: evaluating: boom

## Unsupported, by reason

| Reason | Tests |
|---|---:|
| op TopK | 2 |
| element type DOUBLE | 1 |
| control flow (Loop) | 1 |
