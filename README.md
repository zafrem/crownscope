# crownscope

Ranks Korean blog posts and shopping reviews that look mass-produced with generative AI, and shows the evidence
behind every score. It is a triage filter for human review, not an authorship detector: it never gives a verdict.

🌐 [한국어 README](README.ko.md) · Showcase site (English, 한국어, 日本語, 简体中文, 繁體中文): <https://crownscope.codeforlife.dev> (not published yet)

## Status

Early and **not released**: it is not on PyPI and no license has been chosen yet. Install from source:

```
pip install git+https://github.com/zafrem/crownscope             # core: PyYAML only
pip install "crownscope[embed] @ git+https://github.com/zafrem/crownscope"   # + sentence embeddings
pip install "crownscope[model] @ git+https://github.com/zafrem/crownscope"   # + language-model signals
```

**Nothing has been evaluated on real labeled Korean data.** The weights, thresholds and reference values in the
bundled packs (`blog`, `shopping`) are placeholders. Every figure on the showcase site comes from synthetic sets
used to test the tooling. The only language-model evaluation so far is small and synthetic (`ppl` on Qwen2.5 0.5B
and 1.5B): a weak signal, and on one set it pointed the wrong way.

## Use

```python
from crownscope import Detector

result = Detector(pack="shopping").analyze(text)   # a Result: grade, score, per-signal contributions, evidence
```

```
# rank a CSV; --cumulative adds clusters, re-scoring of earlier documents and a per-author report
crownscope analyze reviews.csv --pack shopping --cumulative --report-by author

# measure a pack on labeled files, removing one signal group at a time
crownscope evaluate labeled.csv --pack shopping --cumulative --ablation

# learn weights and thresholds from labeled files, with cross-validation
crownscope fit labeled.csv --pack shopping --cumulative -o shopping-fitted.yml
```

Output holds IDs, metadata, scores, signals and evidence, never the document text. Korean CSV saved in Excel can be
read with `--encoding cp949`.

## Signals

All ten signals in the specification are built. A signal that cannot run is left out of the average and says why;
it is never counted as zero. Text under 500 characters gets no score unless a duplicate or repeated layout is found.

| Signal | Looks at | Needs | State |
| --- | --- | --- | --- |
| `pattern` | links, phone numbers, promotional phrases, sponsorship disclosures | nothing | built |
| `style` | repetition, vocabulary variety, sentence rhythm, boilerplate, Korean sentence endings, translationese | nothing | built |
| `near_dup` | near-identical documents in the window (MinHash and LSH) | a stream | built |
| `structure` | repeated layout | a stream | built |
| `novelty` | borrowed content, missing facts, optionally paraphrases | a stream, model optional | built |
| `embed_sim` | paraphrased copies, by meaning | a stream, an embedding model | built, not run with a real model |
| `ppl` | how predictable the text is to a language model | a language model | built, weak on a first Korean test |
| `burst` | sentence-to-sentence variation in predictability | a language model | built, not yet evaluated |
| `fast_detect` | probability curvature (Fast-DetectGPT style) | a language model | built, not yet evaluated |
| `binoculars` | observer/performer perplexity ratio | two models, one tokenizer | built, not yet evaluated |

`embed_sim`, `ppl`, `burst`, `fast_detect` and `binoculars` are disabled in the bundled packs. A pack enables one by
listing it in `signals` and `weights` and naming the model(s) in its options; nothing is downloaded or loaded otherwise.

## Translations

The Korean README and the Japanese and Chinese text on the showcase site were written by an AI assistant and have
not been reviewed by native speakers.

Labeled data, tests and the specification live in the private `crownscope-forge` repository.
