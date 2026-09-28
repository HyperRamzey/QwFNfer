# Build the relocatable Windows release bundle:
#   dist/qwfnfer-windows-x64-cuda.zip
#
# The counterpart of scripts/package.sh, and the same bundle it builds: bin/ with the
# engine and the shared libraries it loads, tools/ with the console, qwfnfer.cmd to
# start it, and the README, LICENSE, VERSION and INSTALL.txt beside those. Unzip
# anywhere and run qwfnfer.cmd -- no source tree, no build step, no PATH.
#
#   scripts/package.ps1
#   scripts/package.ps1 -Version v0.2.4 -BuildDir build-verify
#   scripts/package.ps1 -NoCudaRuntime      # bin/ without cublas; the driver and the
#                                           # CUDA 13 redistributables come from the
#                                           # machine instead (they must match the
#                                           # build's CUDA major version, 13)
#
# What the bundle does not carry: the NVIDIA driver, and the model. Everything else the
# engine loads is in bin/, because Windows resolves a DLL beside the executable before it
# looks at PATH -- which is why the libraries go in bin/ and not in a lib/ subdirectory.
param(
    [string]$Version = "",
    [string]$BuildDir = "build-verify",
    [string]$OutDir = "dist",
    [switch]$NoCudaRuntime
)

$ErrorActionPreference = "Stop"
Set-Location (Resolve-Path (Join-Path $PSScriptRoot ".."))
$root = (Get-Location).Path

if (-not $Version) {
    $Version = (& git describe --tags --always --dirty 2>$null)
    if (-not $Version) { $Version = (Get-Date -Format yyyyMMdd) }
}

# The engine plus the tools the console and a hand at the command line use. The unit-test
# and development binaries (qwfn-test-hc, qwfn-refdump, qwfn-logits, qwfn-vtest) are left
# out: they are for working on the engine, not for running it.
$engine = @(
    "qwfn-server.exe",   # the server the console starts and clients talk to
    "qwfn-tok.exe",      # tokenizer
    "qwfn-gen.exe",      # batch generation and the reference replay
    "qwfn-chat.exe",     # interactive terminal client
    "qwfn-inspect.exe",  # what the model actually holds
    "qwfn-io-test.exe"   # the read pattern the drive probe measures
)
# ggml/llama.cpp's own shared libraries, then the CUDA runtime. A zip cannot carry the
# symlinks the .so names are, which is why the Linux bundle copies them -L; on Windows
# these are real files already.
$libs = @("ggml-base.dll", "ggml-cpu.dll", "ggml-cuda.dll", "ggml.dll", "llama.dll", "mtmd.dll")
$cuda = @("cublas64_13.dll", "cublasLt64_13.dll")

$out = Join-Path $root (Join-Path $OutDir "qwfnfer-windows-x64-cuda")
if (Test-Path $out) { Remove-Item -Recurse -Force $out }
New-Item -ItemType Directory -Force -Path (Join-Path $out "bin") | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $out "tools\console") | Out-Null

"== bundle $out"
foreach ($f in $engine) {
    $p = Join-Path $BuildDir $f
    if (-not (Test-Path $p)) { throw "$f is not in $BuildDir -- build the engine first" }
    Copy-Item $p (Join-Path $out "bin")
}
foreach ($f in $libs) {
    $p = Join-Path $BuildDir $f
    if (-not (Test-Path $p)) { throw "$f is not in $BuildDir -- stage llama.cpp's shared libraries first" }
    Copy-Item $p (Join-Path $out "bin")
}
if (-not $NoCudaRuntime) {
    foreach ($f in $cuda) {
        $p = Join-Path $BuildDir $f
        if (-not (Test-Path $p)) { throw "$f is not in $BuildDir (or pass -NoCudaRuntime)" }
        Copy-Item $p (Join-Path $out "bin")
    }
}
Copy-Item "tools\qwfn_console.py" (Join-Path $out "tools")
Copy-Item "tools\qwfn_router.py" (Join-Path $out "tools")
Copy-Item "tools\console\index.html" (Join-Path $out "tools\console")
Copy-Item "scripts\qwfnfer.cmd" $out
Copy-Item "README.md" $out
Copy-Item "LICENSE" $out
Set-Content -LiteralPath (Join-Path $out "VERSION") -Value $Version -NoNewline -Encoding ascii

$cudaLine = if ($NoCudaRuntime) {
    "bin/ carries no CUDA runtime: install the CUDA 13 redistributables (cublas64_13.dll, cublasLt64_13.dll)."
} else {
    "Everything the engine needs is in bin/ except the NVIDIA driver. See README.md."
}
@"
qwfnfer $Version -- Qwen3.8-Flash-Next on one 16 GB GPU (Windows x86-64, NVIDIA)

1. Unzip anywhere and run:   qwfnfer.cmd
   (opens the console at http://127.0.0.1:8090; needs Python 3.10+ and an NVIDIA driver
   new enough for your CUDA 13 build)
2. Get the model once, with the console:  open the console, press "Add model folder",
   point it at your Hugging Face cache, and let it find the shards
   (or download them yourself with hf: pip install -U huggingface_hub)
3. Pick a tier (Chat, Agentic coding, Agentic coding+ or your own Custom one) and press
   Auto-tune & start: the console measures your drive, threads and memory, picks every
   flag, starts the server and verifies it.
   The endpoint is http://127.0.0.1:8080/v1

A Custom tier is where the K and V of the KV cache are named apart, along with the
context, batch, RAM and VRAM reserve, and where the attention state is put in RAM. The
console prices the pair as the mean of the two sides, which is exactly what the engine
allocates for it.

$cudaLine
"@ | Set-Content -LiteralPath (Join-Path $out "INSTALL.txt") -Encoding ascii

"== checks"
# The console finds the engine beside itself in the bundle and not in a build tree, so
# this is the one that would catch a bundle laid out wrongly.
$env:QWFN_SERVER = ""
$found = Push-Location $out; try {
    $c = Get-Content "tools\qwfn_console.py" -Raw
    if ($c -notmatch 'bin", _EXE') { throw "the console does not look in bin/ -- the bundle layout changed" }
    if (-not (Test-Path "bin\qwfn-server.exe")) { throw "bin\qwfn-server.exe is missing" }
} finally { Pop-Location }
"console resolves the bundled engine: ok"

$zip = Join-Path $root (Join-Path $OutDir "qwfnfer-windows-x64-cuda.zip")
if (Test-Path $zip) { Remove-Item -Force $zip }
# bsdtar reads -a as "pick the format from the extension", which is a great deal faster
# than Compress-Archive on the CUDA libraries.
& tar -a -c -f $zip -C (Split-Path $out) (Split-Path $out -Leaf)
if ($LASTEXITCODE -ne 0) { throw "zip failed" }
"{0}  {1:N1} MB" -f (Split-Path $zip -Leaf), ((Get-Item $zip).Length / 1MB)
"built $Version"
