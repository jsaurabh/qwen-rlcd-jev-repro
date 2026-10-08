# Third-party sources

This repository contains original experiment code and synthetic datasets. It does not vendor model weights or the source trees below. Setup commands clone/download the named immutable revisions, whose own licenses continue to apply.

| Project | Role | Pinned revision |
|---|---|---|
| [Qwen-2.5-1B-RLCD](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | Original public shared-prefill/candidate-logit inference path | `2af86848be75847ccb3553b0941cc51d6ef7e4e9` |
| [Bespoke Nimble](https://github.com/bespokelabsai/nimble) | Candidate classification, contrastive data, and evaluation controls | `f136b3f75721fda4ea961f73993cc50b08488835` |
| [JevBench](https://github.com/fstandhartinger/jevbench) | Public benchmark | `5e95f23cbb7be098a9061fea924c4421620ab1a5` |
| [AutoJev](https://github.com/denis-pplx/autojev) | 255-way decision readout and public data builder | `ee63c1515980491a742f0bd0685c8dc5ca1f00c3` |
| [MLX-LM](https://github.com/ml-explore/mlx-lm) | Apple-Silicon model loading and LoRA primitives | versions pinned in `requirements.lock.txt` |

Model repositories are downloaded separately. Review their model cards and licenses before redistribution or commercial use.
