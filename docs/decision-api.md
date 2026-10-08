# Calling and serving decision models

Use **OpenAI Decisions as the primary client contract**, with **TypeSafe System One compatibility** for our existing experiments. Both endpoints below use the same local probabilities and saved calibration temperature. This is interface compatibility for a documented subset, not a claim to reproduce either hosted model.

## Interface comparison

Checked against the official documentation on October 8, 2026:

| Aspect | OpenAI Decisions | TypeSafe System One |
|---|---|---|
| Endpoint | `/v1/decisions` | `/v1/systemone` |
| Shared evidence | `input`: text or supported user messages | `state`: text or structured JSON |
| Question identification | Ordered array, optional `name` | Map keyed by question ID |
| Condition probability | `predicate` → `probability` | `noul` → `noul` |
| Candidates | `choices` array with typed string/boolean `value` and description | `criteria` map with string keys |
| Ordered rubric | `levels` array | `criteria` array |
| Distributions | Array of typed values and probabilities | Map from candidate/level keys to probabilities |
| Refusals | Per-question `refusal` result | Three answer types documented in the HTTP reference |

Sources: [OpenAI Decisions guide](https://developers.openai.com/api/docs/guides/decisions), [OpenAI request/response reference](https://developers.openai.com/api/reference/resources/decisions/methods/create), [TypeSafe HTTP API](https://docs.typesafe.ai/api).

Decisions is a good public-facing default: candidate order and typed values are explicit, and applications can use the same request vocabulary with OpenAI. System One is useful for our existing datasets and structured state. Neither contract requires a different model or a new training run. The bridge translates data structures around our existing readout.

## Client: no additional dependencies

Run Python from the cloned repository root. Clients return dictionaries, preserve native provider errors as `APIError.status`, and do not read `.env` automatically.

```python
from decision_api import DecisionsClient

client = DecisionsClient(base_url="http://127.0.0.1:8000/v1")
result = client.create(
    model="our-autojev",
    input="Two card charges appeared that I did not authorize.",
    questions=[{
        "type": "predicate", "name": "review",
        "instructions": "Does this report suspected unauthorized activity?",
    }],
)
print(result["answers"][0]["probability"])
```

A complete banking example asks a predicate, choice, and score together:

```bash
python -m examples.banking_decisions --provider local
# Paid hosted calls; export the relevant key in your shell first.
python -m examples.banking_decisions --provider openai --model gpt-6-luna
python -m examples.banking_decisions --provider jev --model jev-latest
```

The example reads `DECISION_API_KEY`, `OPENAI_API_KEY`, or `TYPESAFE_API_KEY` according to the provider. Use a pinned Jev model version when comparing benchmark runs. Each invocation prints the actual model ID, all answers/distributions, and native usage. Running these three commands is a smoke comparison, not a labeled accuracy benchmark.

For the official OpenAI service, point `DecisionsClient` at `https://api.openai.com/v1`. It passes native payloads through, including supported image inputs and `safety_identifier`. For TypeSafe's native contract:

```python
import os
from decision_api import SystemOneClient

client = SystemOneClient(api_key=os.environ["TYPESAFE_API_KEY"])
result = client.create(
    model="jev-latest", state={"message": "My card is missing"},
    questions={"review": {"type": "noul", "instructions": "Is human review warranted?"}},
)
```

`SystemOneClient.from_decisions(...)` translates the text subset of a Decisions request to Jev and returns an answer array. It preserves the provider's confidence and usage. Typed boolean choices are given unique aliases so `true` and the string `"true"` cannot collide. Aliased prompts may produce different model behavior; the string-choice path preserves the original candidate keys.

Timeout defaults to 90 seconds. Retries default to zero to avoid accidental repeat charges. `max_retries=2` enables bounded backoff for explicit rate-limit/server errors; ambiguous network failures are not retried. Redirects are refused so bearer credentials are not forwarded to another host. HTTPS is required except for loopback development.

## Serve a saved 27B LoRA on a GPU

Use the existing AutoJev/Perplexity runtime dependencies and pinned `autojev` inference source described in [the reproduction guide](../REPRODUCING.md) and [Perplexity setup](pplx-decider-v1.1.md). A bare install of the HTTP dependencies does not install the model stack.

In the runtime used for the attention experiment:

```bash
cd /path/to/qwen-rlcd-jev-repro
python -m pip install -r decision_api/requirements.txt
export PYTHONPATH="/content/attention-ablation/release/source/src:$PYTHONPATH"
python -m decision_api.server \
  --artifact /content/attention-ablation-results/noncausal_full_attention/selected \
  --model-name our-autojev --batch-size 4
```

Use an existing, fully saved artifact path; `selected` exists only after calibration completes. After runtime loss, restore it from Drive first using [the checkpoint workflow](../REPRODUCING.md). This server loads our LoRA-plus-readout artifacts, not the full Perplexity release format or MLX adapters.

The service binds `127.0.0.1:8000`, exposes `/health`, `/v1/decisions`, and `/v1/systemone`, and advertises its limits in `/health`. Load one model in one process; requests are serialized to avoid overlapping GPU allocations. Use a separate inference runtime or wait until training finishes. Do not load another 27B model on the active training GPU.

For access from another machine, use an authenticated tunnel or TLS reverse proxy. Setting `--host 0.0.0.0` requires `DECISION_API_KEY`; it does not provision TLS or publish a URL. The client runs wherever Python is installed; the server must run beside the GPU and artifact. The server is a research tool, with no production queue, tenancy, or billing layer.

### Supported local subset and probability semantics

- Decisions: text-string input, string instructions, optional unique names, 2–255 string/boolean choices, 2–10 score levels. System One: text/JSON state and structured instructions/criteria. Both accept 1–64 questions and at most 1 MiB per request.
- Images and Decisions message arrays are explicitly rejected locally. Hosted clients can pass their provider's native multimodal payloads directly. No image support is implied for this text-only finetune.
- Scores use the expected **zero-based level index**, not the winning label or an arbitrary numeric range.
- Local confidence uses the published [TypeSafe formulas](https://docs.typesafe.ai/confidence): rescaled top probability for Choice, and normalized spread around the most likely level for Score. It is **not** the probability of correctness or an implementation of an undocumented OpenAI formula. Noul/predicate returns no separate confidence.
- Saved temperature is applied once to the candidate logits. The API does not itself improve calibration. Compare full distributions with held-out NLL/Brier and reliability plots; do not compare provider confidence numbers as if they were interchangeable.
- Multiple questions are batched as separate state/question rows. This implementation repeats state and does **not** share a prefill or KV cache across questions. Token usage is the actual unpadded token count summed across those rows, with zero generated output tokens; provider billing and caching rules may differ.
- Saved attention mode is honored, including the noncausal full-attention hook. The artifact's maximum length is enforced without truncation. Invalid requests return 422; inference failures return 500. The local model has no trained refusal mechanism, so it does not invent refusal outputs.
- Unknown request fields (including local `safety_identifier`) are rejected. This is a deliberately limited compatibility surface, not full official SDK/service parity.

## Verification

```bash
python -m pip install -r decision_api/requirements.txt httpx
python -m unittest discover -s tests -v
```

Contract tests use deterministic probabilities, not a pretend model. They verify both routes, typed values, score arithmetic, invalid payloads, auth, refusals in the Jev bridge, and real loopback HTTP client requests. GPU inference still requires a saved artifact and the matching model environment.

Validation for this change: 11 contract/HTTP tests passed and one live three-question bridge call succeeded against Jev 1.13.0. No live OpenAI call or 27B server inference was run; the GPU was still training.
