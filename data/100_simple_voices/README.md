# 100 Simple Voices

A corpus of 100 distinctive English-language text sources for n-gram and small-language-model experiments, particularly for source-attribution and "broad listening"-style demos.

Each source is ~1.1MB of cleaned text (matching `tinyshakespeare`'s 1,115,394 bytes for fair comparison) from public-domain or openly-licensed material. The corpus is designed for the "small model, recognizable output" use case — pick a source, train a 5-gram backoff model on it, and the resulting generations should be instantly identifiable from one or two sentences.

## Layout

```
data/100_simple_voices/
├── sources/              100 .txt files, named source_<NNN>_<short-name>.txt
├── vocab_stats.json      full per-source statistics
├── compute_stats.py      re-runnable analysis script
└── README.md             this file
```

## Statistics table

Sorted by `demo` score (composite: small vocab × high distinctness × adequate size). Sources at the top are the best fits for "small model that demos recognizably". Column legend follows the table.

```
   #  source                                     bytes     tokens    vocab    bg_typ     ttr    hpx   zipf   wlen   punct      nl    caps     kl    demo
----  ------------------------------------  ----------  ---------  -------  --------  ------  -----  -----  -----  ------  ------  ------  -----  ------
   1  077_nba-play-by-play                   1,115,371    139,993      510     6,814   0.004   0.06   1.73   4.00   0.091   0.018   0.000   3.50    56.1
   2  076_retrosheet-baseball                1,115,390    212,056      865     4,506   0.004   0.01   1.38   3.86   0.048   0.024   0.000   3.06    45.2
   3  093_pubmed-abs                         1,115,317    150,274   11,860    84,269   0.079   0.35   0.81   5.99   0.033   0.003   0.042   1.85    19.7
   4  021_fannie-farmer                      1,115,349    149,713    5,109    49,182   0.034   0.34   1.04   4.70   0.026   0.024   0.008   1.59    18.6
   5  097_cia-world-factbook                 1,110,670    144,335   11,859    57,966   0.082   0.49   0.87   5.53   0.053   0.039   0.032   1.72    18.3
   6  051_us-code-title18                    1,115,337    159,122    5,503    36,948   0.035   0.31   1.09   4.81   0.061   0.014   0.004   1.53    17.8
   7  022_robert-may-accomplisht-cook          807,163    143,683    5,022    37,688   0.035   0.45   1.23   4.14   0.040   0.022   0.001   1.50    17.6
   8  054_federal-register                   1,112,008    160,369    4,617    36,133   0.029   0.29   1.00   5.34   0.051   0.017   0.019   1.40    16.6
   9  052_uspto-patents                      1,114,163    190,311    4,639    45,113   0.024   0.28   1.04   4.61   0.026   0.004   0.003   1.38    16.3
  10  024_international-jewish-cookbook        830,281    150,965    4,009    46,288   0.027   0.29   1.11   4.31   0.028   0.025   0.028   1.33    16.1
  11  098_ietf-rfcs                          1,115,356    165,909    7,365    66,206   0.044   0.32   0.96   4.99   0.055   0.027   0.071   1.41    15.8
  12  059_cfr-title21                        1,115,261    167,191    5,350    42,674   0.032   0.29   1.04   5.19   0.034   0.015   0.014   1.34    15.7
  13  090_arxiv-math-abs                     1,115,353    171,160    8,950    72,843   0.052   0.37   0.91   5.30   0.022   0.006   0.003   1.41    15.5
  14  099_math-reviews                       1,115,248    163,053    9,745    49,901   0.060   0.18   0.89   5.25   0.045   0.006   0.012   1.41    15.3
  15  091_arxiv-physics-abs                  1,115,384    175,057    8,799    77,753   0.050   0.35   0.86   5.07   0.023   0.004   0.017   1.35    14.8
  16  070_reuters-21578                      1,115,132    172,177   11,224    77,167   0.065   0.39   0.93   4.88   0.026   0.022   0.011   1.27    13.6
  17  026_apicius-roman                        667,872    100,111   12,016    57,940   0.120   0.50   0.95   4.69   0.056   0.026   0.236   1.27    13.6
  18  027_sarah-rorer                          436,411     73,842    2,855    23,451   0.039   0.30   1.15   4.43   0.026   0.025   0.017   1.41    13.1
  19  074_bbc-transcripts                    1,115,382    172,943   37,086   129,094   0.214   0.70   0.92   4.64   0.050   0.047   0.081   1.37    13.0
  20  025_whitehouse-cookbook                1,115,385    201,793    6,088    62,214   0.030   0.37   1.08   4.26   0.032   0.023   0.025   1.07    12.3
  21  092_arxiv-cs-abs                       1,115,256    168,230   10,537    84,648   0.063   0.36   0.87   5.47   0.020   0.005   0.010   1.14    12.3
  22  050_scotus-opinions                    1,114,724    167,404   12,424    77,091   0.074   0.43   0.96   4.95   0.044   0.019   0.014   1.15    12.2
  23  057_un-treaties                        1,114,958    174,151    6,649    52,557   0.038   0.30   0.97   5.07   0.023   0.013   0.021   1.06    12.1
  24  010_tinystories                        1,115,275    216,800    5,039    60,002   0.023   0.26   1.12   3.91   0.037   0.012   0.000   1.01    11.9
  25  023_eliza-acton                        1,115,384    184,452    5,370    55,160   0.029   0.33   1.04   4.29   0.031   0.020   0.015   0.99    11.5
  26  000_kjv-bible                          1,115,393    205,740    5,677    49,014   0.028   0.34   1.18   4.04   0.034   0.022   0.012   0.99    11.4
  27  028_mrs-rundell                          454,580     79,611    3,661    30,188   0.046   0.37   1.11   4.06   0.034   0.023   0.002   1.10    10.7
  28  009_lotus-sutra-kern                     867,498    143,667   14,118    68,138   0.098   0.55   0.97   4.56   0.045   0.029   0.011   1.02    10.6
  29  005_book-of-mormon                     1,115,340    204,500    5,027    47,147   0.025   0.31   1.21   4.16   0.028   0.021   0.001   0.91    10.6
  30  081_stackoverflow                      1,114,383    189,613    8,414    83,089   0.044   0.38   0.98   4.52   0.035   0.021   0.010   0.94    10.3
  31  095_wikipedia-sample                   1,115,267    168,691   20,014    99,453   0.119   0.49   0.85   5.05   0.028   0.012   0.009   1.00    10.1
  32  087_reddit-wallstreetbets              1,115,372    198,917   12,747   102,393   0.064   0.49   1.03   4.16   0.025   0.032   0.032   0.94     9.9
  33  029_soyer-housewife                      886,518    155,523    6,972    55,464   0.045   0.41   1.07   4.20   0.038   0.020   0.007   0.85     9.6
  34  039_chekhov                              751,717    137,997    8,262    63,314   0.060   0.43   1.07   4.04   0.056   0.024   0.044   0.80     8.9
  35  006_apocrypha                            795,270    141,746    6,696    54,618   0.047   0.41   1.12   4.21   0.033   0.025   0.000   0.78     8.9
  36  094_simple-wikipedia                   1,115,210    178,753   15,117    95,822   0.085   0.47   0.92   4.86   0.031   0.009   0.003   0.85     8.8
  37  078_historical-obits                   1,114,404    161,884   24,493    91,954   0.151   0.68   0.97   4.38   0.038   0.025   0.007   0.88     8.7
  38  001_quran-rodwell                      1,098,623    201,281    9,885    72,497   0.049   0.42   1.14   4.16   0.036   0.024   0.006   0.80     8.7
  39  038_ibsen                              1,115,327    193,512    8,862    71,826   0.046   0.40   1.18   4.10   0.054   0.037   0.055   0.78     8.6
  40  075_nyt-historical                     1,115,226    182,035   26,319   123,266   0.145   0.60   0.91   4.49   0.044   0.040   0.034   0.88     8.6
  41  056_magna-carta-medieval               1,114,649    190,792   11,626    79,972   0.061   0.43   1.00   4.54   0.031   0.018   0.002   0.80     8.5
  42  080_hn-comments                        1,115,303    185,541   15,004   107,085   0.081   0.46   0.95   4.68   0.035   0.013   0.011   0.81     8.4
  43  020_beeton-household                   1,115,366    183,688   10,323    73,805   0.056   0.42   1.01   4.45   0.042   0.023   0.027   0.77     8.3
  44  032_marlowe                            1,115,384    175,341   14,802   100,170   0.084   0.47   1.02   4.32   0.057   0.025   0.032   0.79     8.2
  45  053_us-founding                        1,114,487    186,444    8,670    69,042   0.046   0.34   1.03   4.84   0.019   0.017   0.011   0.74     8.2
  46  072_chronicling-america                1,115,274    182,150   24,733   125,983   0.136   0.58   0.91   4.44   0.037   0.035   0.074   0.83     8.2
  47  019_oz-baum                              658,664    122,034    6,281    53,087   0.051   0.38   1.05   4.11   0.037   0.023   0.003   0.71     8.1
  48  058_us-constitution-amendments         1,113,878    186,298    8,753    68,931   0.047   0.35   1.03   4.84   0.019   0.017   0.011   0.74     8.1
  49  030_tinyshakespeare                    1,115,388    203,839   12,373   107,203   0.061   0.45   1.07   4.20   0.049   0.036   0.038   0.76     8.1
  50  088_reddit-buyitforlife                1,115,378    203,149   12,508    98,595   0.062   0.45   1.02   4.23   0.026   0.014   0.006   0.76     8.0
  51  049_keats                                557,004     91,534   10,573    59,391   0.116   0.45   0.97   4.36   0.042   0.024   0.007   0.81     8.0
  52  055_blackstone-commentaries            1,114,371    191,109   11,474    81,599   0.060   0.43   1.02   4.53   0.037   0.018   0.007   0.75     8.0
  53  013_grimm-fairy-tales                    530,834    102,114    4,746    39,720   0.046   0.36   1.12   3.88   0.031   0.017   0.005   0.68     8.0
  54  085_reddit-amitheasshole               1,115,352    207,021    8,824    80,968   0.043   0.42   1.08   4.15   0.025   0.011   0.015   0.72     8.0
  55  048_coleridge                          1,112,931    163,229   15,279    94,869   0.094   0.46   0.96   4.32   0.054   0.027   0.016   0.76     7.9
  56  086_reddit-relationships               1,115,313    205,633    9,004    81,119   0.044   0.42   1.12   4.20   0.025   0.011   0.003   0.72     7.9
  57  096_britannica-1911                    1,115,304    181,879   17,781    95,859   0.098   0.49   0.91   4.76   0.037   0.016   0.008   0.77     7.9
  58  037_moliere                            1,106,513    201,312   11,244    87,049   0.056   0.44   1.07   4.08   0.046   0.027   0.055   0.74     7.9
  59  044_rossetti-christina                   906,638    153,861    9,102    67,604   0.059   0.29   1.01   4.08   0.036   0.033   0.008   0.72     7.9
  60  073_ap-wire-vintage                    1,114,345    165,615   13,262    85,164   0.080   0.45   0.95   4.65   0.021   0.019   0.015   0.73     7.7
  61  031_shakespeare-complete               1,115,381    203,347   11,974   104,335   0.059   0.46   1.09   4.09   0.042   0.037   0.041   0.72     7.7
  62  071_newsgroups-20                      1,115,199    181,393   19,620   103,473   0.108   0.52   1.00   4.29   0.057   0.021   0.025   0.75     7.6
  63  079_tabloid-headlines                  1,114,915    192,536   22,920   114,321   0.119   0.58   0.95   4.36   0.038   0.031   0.041   0.77     7.6
  64  083_stackexchange-cook                 1,115,287    196,348   11,099    96,517   0.057   0.42   0.98   4.41   0.034   0.013   0.002   0.71     7.6
  65  034_sophocles                          1,115,387    194,833   12,840   106,354   0.066   0.43   1.05   4.18   0.042   0.031   0.024   0.72     7.6
  66  089_reddit-legaladvice                 1,113,778    199,413    9,435    83,140   0.047   0.41   1.05   4.35   0.024   0.012   0.004   0.69     7.5
  67  035_euripides                          1,115,386    192,621   12,021    94,618   0.062   0.44   1.04   4.20   0.042   0.023   0.023   0.69     7.4
  68  063_austen                             1,115,331    197,094    8,132    78,868   0.041   0.37   1.10   4.33   0.029   0.019   0.002   0.66     7.3
  69  033_aeschylus                          1,115,382    191,760   14,982   110,248   0.078   0.44   0.99   4.32   0.037   0.026   0.019   0.70     7.3
  70  036_aristophanes                       1,111,314    193,127   13,171    98,641   0.068   0.40   1.07   4.38   0.048   0.026   0.040   0.67     7.1
  71  046_longfellow                         1,115,241    196,290   14,032   103,204   0.071   0.41   0.94   4.33   0.038   0.030   0.009   0.68     7.1
  72  062_mark-twain                         1,115,364    214,301   11,404    86,614   0.053   0.43   1.15   3.83   0.030   0.021   0.002   0.65     6.9
  73  082_stackexchange-eng                  1,115,390    189,117   15,254    97,416   0.081   0.48   1.01   4.48   0.042   0.019   0.003   0.63     6.5
  74  040_whitman-leaves-grass               1,115,298    190,519   14,030    85,482   0.074   0.35   1.01   4.33   0.038   0.022   0.003   0.62     6.5
  75  065_burroughs-er                       1,115,368    201,553   11,051    90,955   0.055   0.37   1.01   4.33   0.020   0.021   0.002   0.59     6.4
  76  066_wilkie-collins                     1,115,366    203,575    8,956    77,597   0.044   0.37   1.11   4.16   0.029   0.019   0.002   0.58     6.4
  77  043_tennyson                           1,111,209    195,648   13,949   101,774   0.071   0.45   1.01   4.13   0.035   0.027   0.000   0.60     6.3
  78  084_reddit-eli5                        1,115,202    196,235   12,612    97,493   0.064   0.43   0.99   4.43   0.026   0.012   0.004   0.59     6.2
  79  067_stephen-crane                      1,115,359    200,428   13,812    98,612   0.069   0.41   1.03   4.30   0.035   0.021   0.001   0.59     6.2
  80  060_lovecraft                          1,115,329    188,489   16,842   108,372   0.089   0.42   0.96   4.69   0.025   0.017   0.002   0.60     6.1
  81  069_hg-wells                           1,115,356    200,648   12,648    93,531   0.063   0.41   1.04   4.28   0.028   0.019   0.004   0.57     6.1
  82  047_wordsworth                         1,115,351    171,518   14,976    99,189   0.087   0.42   0.97   4.45   0.043   0.025   0.011   0.57     5.9
  83  016_carroll-alice                        323,613     58,521    3,863    26,900   0.066   0.39   1.12   3.92   0.033   0.024   0.002   0.82     5.8
  84  068_owen-wister                        1,115,334    201,670   12,064    95,039   0.060   0.42   1.08   4.14   0.031   0.021   0.001   0.54     5.8
  85  061_doyle-sherlock                     1,115,334    195,561   10,539    82,062   0.054   0.40   1.09   4.09   0.027   0.021   0.001   0.50     5.4
  86  064_dickens                            1,115,370    201,025   12,085    92,050   0.060   0.42   1.09   4.21   0.033   0.020   0.002   0.49     5.2
  87  045_poe-poems                            954,297    153,287   13,159    82,823   0.086   0.44   0.99   4.44   0.030   0.019   0.003   0.49     5.2
  88  014_andersen-fairy-tales                 303,342     56,115    5,194    29,988   0.093   0.47   1.09   4.11   0.033   0.019   0.002   0.67     4.4
  89  007_confucius-analects                   168,565     29,124    3,148    15,137   0.108   0.44   1.07   4.22   0.052   0.019   0.032   1.17     4.2
  90  018_beatrix-potter                       163,269     28,634    3,518    16,606   0.123   0.44   0.99   4.34   0.047   0.032   0.018   1.20     4.2
  91  012_aesop-fables                         244,736     45,078    5,442    25,076   0.121   0.48   1.00   4.18   0.025   0.021   0.010   0.80     4.2
  92  015_lear-nonsense                        215,384     33,147    4,630    17,899   0.140   0.47   1.01   4.25   0.040   0.033   0.016   1.05     4.1
  93  008_methodist-hymnal                      54,796      6,763    1,315     4,229   0.194   0.52   1.09   4.15   0.046   0.032   0.035   3.57     3.4
  94  002_bhagavad-gita-arnold                 124,090     20,602    0.97   3,735    15,591   0.181   0.54   0.97   4.44   0.043   0.026   0.013   1.26     3.1
  95  041_dickinson                            174,661     30,913    5,789    23,369   0.187   0.57   0.97   4.27   0.043   0.048   0.025   0.82     2.9
  96  011_mother-goose                         116,428     17,983    2,492    10,761   0.139   0.43   0.98   3.98   0.049   0.036   0.024   1.26     2.9
  97  042_blake-songs                           97,096     17,096    2,469     9,225   0.144   0.37   0.95   4.12   0.035   0.034   0.018   1.09     2.4
  98  004_dhammapada                            66,763     11,784    1,841     7,043   0.156   0.49   1.05   4.25   0.040   0.024   0.002   1.32     2.1
  99  003_tao-te-ching-legge                    59,038     10,412    1,929     6,893   0.185   0.55   1.04   4.27   0.041   0.023   0.001   1.05     1.4
 100  017_stevenson-childs                      52,253      8,236    1,828     6,194   0.222   0.56   1.00   3.96   0.029   0.033   0.010   1.16     1.3
```

### Column legend

| col | meaning |
|---|---|
| `bytes` | UTF-8 byte count of the source file |
| `tokens` | word-token count after lowercasing on `/[a-zA-Z]+(?:'[a-zA-Z]+)?/` |
| `vocab` | distinct word types |
| `bg_typ` | distinct word-bigram types (relevant for n-gram model size) |
| `ttr` | type-token ratio = vocab / tokens. Lower = more repetitive, more compressible for n-gram modeling |
| `hpx` | hapax-legomena fraction (vocab seen exactly once). Lower = denser word reuse |
| `zipf` | slope of log(freq) vs log(rank) over top-1000 words. ~1.0 is canonical Zipfian; higher = vocabulary even more skewed toward function words |
| `wlen` | mean word length in chars. Larger = more technical or compounding vocab |
| `punct` | punctuation density (chars in `.,;:!?-()[]"'` / total chars) |
| `nl` | newline density (often a verse / line-structured-format signal) |
| `caps` | fraction of tokens that are ≥3-char ALL-CAPS words (legal docs, RFCs, sermons, news) |
| `kl` | KL divergence vs combined-baseline word distribution. Higher = the source's vocabulary is more distinctive from the corpus average |
| `demo` | composite score: `(1 / log(vocab+1)) × kl × min(1, tokens/100K)`. Higher = better fit for "train a small model that demos recognizably" |

## Notable observations

- **Sports play-by-play has shockingly small vocab.** NBA at 510 words, Retrosheet baseball at 865. Most of the text is player names plus a fixed lexicon of ~30 action verbs ("makes", "misses", "rebound", "second base"). They top the demo ranking by a wide margin.
- **Specialized registers dominate the top of the rankings.** Cookbooks, patents, federal regulations, RFC technical specs, scientific abstracts, US Code — these all have small concentrated vocabularies that don't overlap with literary English. Each one has a "diagnostic vocabulary" (cookbook: `marguerites, pekoe, cottolene`; USPTO: `flanch, mandrel, journaled`; arXiv math: `hyperkaehler, varepsilon, pezzo`) that makes their output instantly recognizable.
- **Classic novels rank LOW.** Austen (#68), Mark Twain (#72), Wilkie Collins (#76), Dickens (#86). They all share Victorian/19th-century novelistic vocabulary and aren't very distinctive from each other. The literature in this corpus is mostly *not* what you'd pick for a small-model demo.
- **Religious & sacred texts hit real public-domain ceilings.** Bhagavad Gita, Tao Te Ching, Dhammapada, Confucius Analects, Methodist Hymnal are inherently smaller than 1.1MB. They have high distinctness but the size term hurts the composite. Methodist Hymnal has the highest KL in the whole corpus (3.57) but only 6,763 tokens.
- **Verse anthologies are similar.** Mother Goose, Lear, Blake, Stevenson are tiny relative to the demo's preferred size, despite high per-token distinctness.

## Acquisition notes (caveats)

**Substitutions.** Roughly 30% of the originally-supplied Project Gutenberg IDs in the agent prompts were wrong (pointed at unrelated works). Agents corrected via PG search and substituted equivalent works where the originals weren't on PG. Examples:

- 006 apocrypha: substituted PG 124 (Deuterocanonical Books); original ID was a Japanese plays anthology
- 022 cookbook: Hannah Glasse not on PG; substituted Robert May's 1685 *The Accomplisht Cook*
- 024 cookbook: Settlement Cook Book not on PG; substituted *International Jewish Cookbook* (1918)
- 032 marlowe: agent concatenated 7 Marlowe plays after originally-supplied IDs were partly wrong
- 060 lovecraft: PG 32032 turned out to be Philip K. Dick, not Lovecraft; correctly excluded
- 073 ap-wire-vintage: no PD source for AP wire; substituted *Source Records of the Great War* (1923)
- 074 bbc-transcripts: BBC archive copyrighted; substituted *Radio Times 1924-26* (BBC's own programming guide)
- 075 nyt-historical: NYT paywalled; substituted IA-hosted NYT 1910-1923 PD issues
- 096 britannica-1911: originally-supplied IDs were *La Vita Nuova*; agent found real Britannica 11th ed volumes

All substitutions are real public-domain text in the same register/era as the original target. See per-bucket completion reports for full provenance.

**OCR artifacts inflate apparent vocab in 4 sources:**

- 009 lotus-sutra-kern: archive.org djvu OCR
- 072 chronicling-america: LoC newspaper OCR
- 074 bbc-transcripts: IA Radio Times OCR
- 079 tabloid-headlines: IA Police Gazette OCR

Look for OCR-style "distinctive words" like `theit, broadeasting, lonion` (BBC) or fragmented chunks. The reported vocab counts for these sources are inflated; a second OCR-cleaning pass would shrink them substantially.

**Acquisition pivots worth knowing for re-runs:**

- **arXiv main API is IP-rate-limited (Fastly hard 429).** Use `oaipmh.arxiv.org/oai` instead; supports resumption tokens, no rate cliff.
- **Reddit's own JSON endpoints all return 403 unauth in 2026.** Pushshift mirror dead. Working replacement: `arctic-shift.photon-reddit.com/api/comments/search`.
- **Chronicling America has aggressive Cloudflare bot mitigation** after ~20 requests. Pace at 1 req/5s or use bulk batch downloads.
- **UN.org returns 403 Cloudflare.** Yale Avalon Project (`avalon.law.yale.edu`) has UN Charter + 70+ 20th-century treaties without the bot wall.
- **Wikipedia REST API blanket-429s at 8-thread parallelism.** Use the multistream bz2 dump for Simple Wikipedia and the action API single-thread for regular Wikipedia.

## License & provenance

All sources are public-domain or openly-licensed for research/derivative use:

- **Project Gutenberg** (most religious texts, drama, poetry, classic novels, cookbooks pre-1928, Apicius, Britannica 1911): public domain
- **HuggingFace `roneneldan/TinyStories`** (Eldan & Li): permissive research use
- **US government bulk data** (USPTO, US Code Title 18, CFR Title 21, Federal Register, SCOTUS opinions, US founding documents, CIA World Factbook 2010): public domain by US federal works rule
- **Internet Archive PD originals** (LA Times 1911-16, NYT 1910-23, Police Gazette 1886-1905, Radio Times 1924-26, US patents 1830s-1900s, Annual Register obits): pre-1929 US publications, public domain
- **Arctic-Shift Reddit dump, HN Firebase API, Stack Exchange API**: subject to the original platforms' terms
- **arXiv abstracts via OAI-PMH** (math/physics/CS): permissive research/derivative use
- **Yale Avalon Project** (UN treaties): academic/research use
- **Karpathy `char-rnn` tinyshakespeare**: MIT (Karpathy's repo)

When redistributing this corpus, preserve the per-source attribution implied by the filenames and document any further transformations.

## Reproducing the statistics

```bash
cd /Users/atrask/Desktop/abcGPT/data/100_simple_voices
python3 compute_stats.py
```

Regenerates `vocab_stats.json` and prints the ranked table. No external dependencies beyond Python 3 stdlib.

## Sources, organized by acquisition bucket

| bucket | range | theme | notes |
|---|---|---|---|
| 1 | 000-009 | religious & sacred | KJV Bible, Quran (Rodwell), Bhagavad Gita (Arnold), Tao Te Ching (Legge), Dhammapada, Book of Mormon, Apocrypha (Deuterocanonical Books), Confucius Analects, Methodist (Indian) Hymnal, Lotus Sutra (Kern via archive.org) |
| 2 | 010-019 | children's lit & verse | TinyStories, Mother Goose, Aesop, Grimm, Andersen, Lear nonsense, Carroll/Alice, Stevenson's *Child's Garden*, Beatrix Potter (18 tales), Wizard of Oz + sequels |
| 3 | 020-029 | cookbooks | Beeton, Fannie Farmer, Robert May 1685, Eliza Acton, International Jewish Cookbook, White House Cookbook, Apicius, Sarah Rorer (6 pamphlets), Mrs. Rundell, Soyer's Modern Housewife |
| 4 | 030-039 | drama | tinyshakespeare, Shakespeare complete, Marlowe (7 plays), Aeschylus (7 sources), Sophocles (6), Euripides (4), Aristophanes (Eleven Comedies), Molière (13 plays), Ibsen (7), Chekhov |
| 5 | 040-049 | poetry | Whitman's *Leaves of Grass*, Dickinson (Series I-III), Blake (*Songs*, *Marriage*), Tennyson, Christina Rossetti, Poe poems, Longfellow, Wordsworth, Coleridge, Keats |
| 6 | 050-059 | legal & government | SCOTUS opinions (2022 term), US Code Title 18 (Crimes), USPTO patents (19th-c via IA), US Founding Documents (Declaration + Federalist), Federal Register rules, Blackstone + Hobbes, Magna Carta + medieval English law, UN treaties via Yale Avalon, US Constitution + Amendments + Federalist, CFR Title 21 |
| 7 | 060-069 | distinctive authors | HP Lovecraft (14 stories), Sherlock Holmes (Doyle), Mark Twain (Huck + Tom + Pudd'nhead), Austen (P&P + Emma), Dickens (Tale + Bleak + Twist), Edgar Rice Burroughs, Wilkie Collins (Moonstone + Woman), Stephen Crane (multiple), Owen Wister (Virginian + Lady B), HG Wells (TM + WotW + IM + Moreau + extras) |
| 8 | 070-079 | news & journalism | Reuters 21578, 20 Newsgroups, Chronicling America (IA-hosted LA Times 1911-16), Source Records of the Great War (1923), Radio Times 1924-26 (BBC programming guide), NYT 1910-23 (IA), Retrosheet baseball play-by-play (2023), basketball-reference NBA play-by-play (2023-24), Annual Register obituaries, Police Gazette tabloid 1886-1905 |
| 9 | 080-089 | modern internet & casual | Hacker News comments, StackOverflow, english.stackexchange, cooking.stackexchange, Reddit r/eli5, r/AmItheAsshole, r/relationships, r/wallstreetbets, r/BuyItForLife, r/legaladvice |
| 10 | 090-099 | scientific & reference | arXiv math abstracts, arXiv physics abstracts, arXiv CS abstracts, PubMed cancer abstracts, Simple Wikipedia, regular Wikipedia random sample, Britannica 1911 (vol with "Austria, Lower" to "Bacon"), CIA World Factbook 2010, ~90 IETF RFCs, zbMATH Open math reviews |
