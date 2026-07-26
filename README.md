...brief description

## Installation

## Usage

Run the benchmark sequence on a single model...

```bash
llama-tuna -m ~/models/llama-3.1-8B.gguf
```

...or a directory with multiple models.

```bash
llama-tuna -md ~/models
```

By default `llama-tuna` tests the models near their maximum context size, which can be very slow.
It's best to specify your desired context size explicitly.

```bash
llama-tuna -m ~/models/llama-3.1-8B.gguf -c 16384
```

Found parameters can be output as either a llama-server CLI string...

```bash
llama-tuna -m ~/models/llama-3.1-8B.gguf -c 16384 -o cli
```
```text
# output
llama-server -fa on -ctk f16 -ctv f16 -t 4 -ngl 32 -b 2048 -ub 512 -c 16384
```

...or text that can be piped to a models.ini file.

```bash
llama-tuna -m ~/models/llama-3.1-8B.gguf -c 16384 -o ini > models.ini
```
```ini
# models.ini
[llama-3.1-8B]
model = ~/models/llama-3.1-8B.gguf
fa = on
ctk = f16
ctv = f16
t = 4
ngl = 32
b = 2048
ub = 512
c = 16384
```

## CLI Reference

```text
usage: llama-tuna [-h] [-m filename | -md directory] [-fa {on,off}]
                  [-ctk {f16,q8_0,q4_0}] [-ctv {f16,q8_0,q4_0}] [-c context]
                  [-b batch-size] [-ub ubatch-size] [-o {cli,ini}]
                  [--llama-bench path] [-t threads] [-ngl n-gpu-layers]

options:
  -h, --help            show this help message and exit

global settings:
  -m filename           GGUF model file
  -md directory, --models-dir directory
                        directory containing GGUF models
  -fa {on,off}          use flash attention
  -ctk {f16,q8_0,q4_0}  cache type K
  -ctv {f16,q8_0,q4_0}  cache type V
  -c context            desired context length, default: model max
  -b batch-size         batch size
  -ub ubatch-size       microbatch size
  -o {cli,ini}          output format: llama-server CLI string or ini
  --llama-bench path    path to llama-bench binary

test parameters:
  if supplied, will not be searched for

  -t threads            number of threads for CPU inference
  -ngl n-gpu-layers     number of layers offloaded to GPU
```

## Environment Variables

Some options can be provided by setting an environment variable. CLI options take precedence if both are provided

| Variable | Description |
|---|---|
| `LLAMA_BENCH` | path to the `llama-bench` binary |
| `MODELS_DIR` | directory containing GGUF models |

## Limitations

- Only GGUF v[2,3] models are supported.
- ...
