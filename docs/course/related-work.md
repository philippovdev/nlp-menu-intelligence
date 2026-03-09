# Related Work

Exact public work on `menu item text -> fixed category label` is sparse. The
closest published evidence splits into three buckets:

- short text and product title categorization
- food, menu, and recipe understanding
- structured extraction from restaurant or receipt-like text

The strongest domain-near paper I found is
[Latif et al., 2025](https://aclanthology.org/2025.inlg-main.31/), but it
studies menu normalization and clustering rather than supervised menu category
classification. That gap should be stated explicitly in the final report.

## Short Text and Product Categorization

- [Joulin et al., 2017](https://aclanthology.org/E17-2068/)
  `joulin-etal-2017-bag`
  Task: efficient short text classification and large-output tag prediction.
  Datasets: eight sentiment or topic benchmarks including AG, Sogou, DBPedia,
  Yelp, Yahoo, and Amazon datasets. Reported result: `fastText` with bigrams
  reaches `92.5%` test accuracy on AG and `98.6%` on DBPedia, while the paper
  also reports training on more than one billion words in under ten minutes on
  CPU. Why it matters: it is the clearest primary-source justification for
  strong CPU-friendly linear baselines before moving to heavier models.
  Directly comparable: no.

- [Xia et al., 2017](https://aclanthology.org/E17-2105/)
  `xia-etal-2017-large`
  Task: product title categorization in e-commerce.
  Dataset: millions of Japanese product titles mapped to `35` product
  categories. Reported result: their attention CNN maintains `>96%` accuracy
  while cutting training time from three weeks to three days compared with the
  previous gradient-boosted tree setup. Why it matters: this is a close analog
  to short catalog-title classification under a fixed taxonomy. Directly
  comparable: no.

- [Cevahir and Murakami, 2016](https://aclanthology.org/C16-1051/)
  `cevahir-murakami-2016-large`
  Task: large-scale hierarchical product categorization from title and
  description.
  Dataset: around `150` million products with a taxonomy of up to `28,338`
  leaf categories. Reported result: first predictions match `81%` of merchant
  assignments when `others` categories are excluded. Why it matters: it shows
  how product-title style categorization behaves once taxonomy size becomes very
  large. Directly comparable: no.

## Food, Menu, and Recipe Understanding

- [Wiegand and Klakow, 2013](https://aclanthology.org/I13-1003/)
  `wiegand-klakow-2013-towards`
  Task: contextual healthiness classification of food-item mentions.
  Dataset: `2,440` manually annotated German forum instances from
  `chefkoch.de`. Reported result: the best feature set reaches `53.9` F1 for
  `Is-Healthy` and `51.0` F1 for `Is-Unhealthy`. Why it matters: it is a real
  food-text classification paper on short mentions, but the labels are about
  healthiness rather than menu taxonomy categories. Directly comparable: no.

- [Hu et al., 2023](https://aclanthology.org/2023.emnlp-main.924/)
  `hu-etal-2023-diner`
  Task: dish name recognition from recipe instructions.
  Dataset: `3,811` dishes and `228,114` recipes in the DiNeR benchmark.
  Reported result: the strongest T5-based setup reaches `59.2` F1 and `23.2`
  exact match on the out-of-distribution TMCD split, and `84.4` F1 / `60.2`
  exact match on the random split. Why it matters: it is food-domain text
  understanding with compositional dish names, which is relevant to noisy menu
  naming and normalization. Directly comparable: no.

- [Latif et al., 2025](https://aclanthology.org/2025.inlg-main.31/)
  `latif-etal-2025-restaurant`
  Task: large-scale restaurant menu normalization and clustering.
  Dataset: proprietary collection of more than `700,000` unique menu items.
  Reported result: CPSAF improves intra-cluster similarity from `0.88` to
  `0.98`, reaches `100%` cluster coverage, and reduces singleton clusters by
  `33%`. Why it matters: this is the closest menu-domain paper I found, but it
  optimizes clustering and normalization, not supervised fixed-label
  classification. Directly comparable: no, but it is the closest domain match.

## Information Extraction and Structured Extraction

- [Peng et al., 2017](https://aclanthology.org/E17-1043/)
  `peng-etal-2017-may`
  Task: structured restaurant-order extraction from dialogue.
  Dataset: restaurant order-taking conversations paired with structured order
  records.
  Reported result: the best neural attention model with memory gate and role
  markers reaches `71.2` F1 and `51.8` whole-order accuracy on the human
  transcript test set, and `65.7` F1 / `45.9` accuracy on the ASR test set.
  Why it matters: it is restaurant-domain structured extraction, but from
  conversation rather than menu-line text. Directly comparable: no.

- [Jaume et al., 2019](https://guillaumejaume.github.io/FUNSD/)
  `jaume2019funsd`
  Task: form understanding with entity labeling and linking.
  Dataset: `199` fully annotated forms, `31,485` words, `9,707` semantic
  entities, and `5,304` relations.
  Reported result: the site presents the dataset as a benchmark resource rather
  than a single headline score. Why it matters: FUNSD is a strong analog for
  field extraction and relation linking once menu documents or OCR outputs are
  represented as semi-structured forms. Directly comparable: no.

- [Park et al., 2019](https://openreview.net/forum?id=SJl3z659UH)
  `park2019cord`
  Task: post-OCR receipt parsing.
  Dataset: more than `11,000` Indonesian receipts with OCR annotations and
  multi-level parsing labels; the paper describes eight superclasses and `54`
  subclasses overall.
  Reported result: the paper is primarily a dataset contribution and does not
  foreground one headline benchmark score. Why it matters: it is especially
  relevant to this project because its schema includes menu-name, quantity, and
  price-like fields after OCR. Directly comparable: no.

- [Huang et al., 2019](https://doi.org/10.1109/ICDAR.2019.00244)
  `huang2019icdar2019`
  Task: scanned receipt OCR and key information extraction.
  Dataset: `1,000` receipts with `600` train/validation and `400` test images.
  Reported result: the competition report states that information extraction
  still leaves clear room for improvement relative to text localization and OCR.
  Why it matters: SROIE is a standard receipt-extraction benchmark and a useful
  analog for price and field extraction under OCR noise. Directly comparable:
  no.

## Comparison Matrix

| Reference key | Approach | Task | Dataset | Metric | Reported result | Relevance to this project | Directly comparable? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `joulin-etal-2017-bag` | fastText linear text classifier | short text classification | AG / Sogou / DBPedia / Yelp / Yahoo / Amazon benchmarks | Accuracy, runtime | `92.5%` on AG with bigrams; CPU training on `>1B` words in `<10` min | Strong justification for simple CPU-first text baselines | No |
| `xia-etal-2017-large` | attention CNN for product titles | product categorization | millions of Japanese titles, `35` categories | Accuracy | `>96%` accuracy | Close analog to short catalog-title classification | No |
| `cevahir-murakami-2016-large` | deep belief nets + autoencoders | hierarchical product categorization | `150M` products, `28,338` leaf categories | first-prediction accuracy | `81%` excluding `others` | Large-taxonomy analog | No |
| `wiegand-klakow-2013-towards` | feature-based classifier | food-item healthiness classification | `2,440` German forum instances | F1 | `53.9` HLTH / `51.0` UNHLTH | Food-text classification with short mentions | No |
| `hu-etal-2023-diner` | T5-based dish name recognition | recipe-to-dish-name prediction | DiNeR: `3,811` dishes, `228,114` recipes | F1, exact match | `59.2` F1 / `23.2` EM on OOD TMCD split | Food-domain lexical composition analog | No |
| `latif-etal-2025-restaurant` | CPSAF hybrid clustering | menu normalization and grouping | `>700,000` menu items | intra-cluster similarity, coverage | `0.98` intra-cluster similarity, `100%` coverage | Closest menu-domain paper, but clustering rather than fixed-label classification | No |
| `peng-etal-2017-may` | seq2seq with memory gate and role markers | structured restaurant-order extraction | restaurant conversations -> order records | item F1, whole-order accuracy | `71.2` F1 / `51.8` accuracy on transcript test | Restaurant-domain extraction analog | No |
| `jaume2019funsd` | FUNSD benchmark | form entity labeling and linking | `199` forms, `31,485` words | entity labeling/linking metrics | dataset paper; no single headline score on site | Strong analog for field extraction from semi-structured text | No |
| `park2019cord` | CORD dataset paper | post-OCR receipt parsing | `>11,000` Indonesian receipts, 8 superclasses, 54 subclasses | dataset/resource paper | no single headline benchmark score in the paper overview | Strong analog for OCR-to-fields parsing with menu-like price lines | No |
| `huang2019icdar2019` | SROIE competition report | scanned receipt OCR + key information extraction | `1,000` receipts | task metrics by challenge track | challenge report highlights that key information extraction remains materially harder than OCR/localization | Standard receipt IE benchmark for noisy semi-structured text | No |

## Practical Takeaways For This Project

- There is strong published support for starting with simple text baselines on
  short inputs before moving to heavier models.
- Product title categorization is the closest mature classification literature,
  but its taxonomies and metadata assumptions are different from menu data.
- Food-domain text work exists, but public work on fixed-label menu item
  categorization is still thin.
- The best public analogs for slot extraction come from receipt and form
  understanding benchmarks rather than menu datasets.
- The literature gap is not classification in general; it is the combination of
  short menu text, fixed category labels, OCR noise, and structured field
  extraction in one public benchmark.
