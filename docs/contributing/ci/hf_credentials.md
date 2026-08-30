# Hugging Face Credentials

Some tests and examples load checkpoints from gated Hugging Face repositories.
This document explains how to supply credentials locally and how CI provides
them. For running tests generally, see
[Test Execution Guide](./test_execution_guide.md). For the full environment
variable inventory, see
[Environment Variables](../../configuration/environment_variables.md).

## When you need credentials

A repository is gated when its owner requires each user to accept a license
before downloading. `stabilityai/stable-audio-open-1.0`, used by the diffusion
offloader tests, is one example.

Without valid credentials the affected tests skip rather than fail. Three
conditions must all hold:

1. `HF_TOKEN` is set in the environment.
2. The token is valid.
3. The account owning the token has accepted the model license on huggingface.co.

Condition 3 is per-account and is the one most often missed: a valid token
alone does not grant access to a gated repository.

## Local setup

Authenticate once with the Hugging Face CLI, which stores the token under
`$HF_HOME`:

```bash
hf auth login
```

`HF_TOKEN` is read first; `HUGGINGFACE_HUB_TOKEN` is honored as a fallback.

Never pass a token value on the command line or paste it into issues, pull
requests, or logs. `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN` are marked for
redaction in the environment variable inventory. If a token is exposed, revoke
it under **Settings -> Access Tokens** on huggingface.co and issue a new one.

## Requesting access to a gated repository

1. Open the model page on huggingface.co while signed in.
2. Accept the license or submit the access request form.
3. Wait for approval. Some repositories grant automatically, others are manual.
4. Re-run the test. The skip disappears once access is granted.

## How CI supplies credentials

CI passes `HF_TOKEN` to test steps as an environment variable. CUDA and AMD
agents declare it through `.buildkite/common/ci_mirror_hardwares.yml`. NPU
pipelines thread it into each step in `.buildkite/npu/test-npu-nightly.yml`.

Authentication failures are not retried. `hub_prefetch` detects them and fails
immediately with an explicit hint, since gating never resolves on retry.

## Troubleshooting

| Message | Meaning |
| --- | --- |
| `gated HF repo ... inaccessible to the current HF_TOKEN` | The token is missing, invalid, or its account has not accepted the license. |
| `HF repo ... not found` | The repository id is wrong, or the repo is private and invisible to this account. |
