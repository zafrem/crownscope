# crownscope

Triage filter that ranks text likely to be mass-produced with generative AI and shows why.
Korean blog and shopping content first. Not an authorship detector; output is for human review.

```
pip install crownscope
```

```python
from crownscope import Detector
result = Detector(pack="shopping").analyze(text)
```

Status: skeleton (SRS phase 1). Labeled data and tests live in the private `crownscope-forge` repository.
