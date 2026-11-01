# Sentiment Error Analysis & Failure Mode Categorization (SW-06)

This document provides a systematic diagnostic audit of 35 real misclassifications extracted from the 1,000-sample test evaluation across both the Baseline (TF-IDF + Logistic Regression) and Deep Learning (DistilBERT SST-2) models.

---

## 1. Quantitative Breakdown of Failure Modes

Across 35 analyzed misclassification cases, failures cluster into four primary linguistic categories:

| Failure Category | Occurrences in Sample (N=35) | Percentage | Primary Vulnerable Model | Root Linguistic Cause |
|---|---|---|---|---|
| **Mixed Sentiment (Contrasting Clauses)** | 13 | 37.1% | Both Baseline & DistilBERT | Co-occurrence of strong positive and negative clauses (e.g. "Loved the book, hated the movie adaptation"). |
| **Negation & Scope Inversion** | 10 | 28.6% | Baseline (TF-IDF + LogReg) | Inability of n-gram bag-of-words to bind negation particles ("not worth", "didn't hold up") across clauses. |
| **Sarcasm & Rhetorical Figures** | 7 | 20.0% | Both Models | Surface polarity is positive ("wonderful phone") but pragmatic intent is sharply critical ("heats up like a stove"). |
| **Text Length / Structural Asymmetry** | 5 | 14.3% | DistilBERT (Truncation) & Baseline (Sparsity) | Reviews exceeding 256 tokens or ultra-terse 3-word fragments where context is insufficient. |

---

## 2. Categorized Case Studies (30+ Authentic Samples)

### Category A: Negation Failures (10 Cases)

1. **Test Item #14**: *"Not worth the $2.99. I bought these for use on my ipod earbuds. I had an old pair that worked great, but..."*
   - *True Label*: NEGATIVE
   - *Baseline*: POSITIVE (Missed "Not worth", dominated by "worked great")
   - *DistilBERT*: NEGATIVE (Correctly bound negation particle)
2. **Test Item #49**: *"These are not the answer to Jawbone Noise Assassin. While Jawbone makes a great ear bud, it doesn't interface..."*
   - *True Label*: NEGATIVE
   - *Baseline*: POSITIVE (Triggered by "great ear bud")
   - *DistilBERT*: NEGATIVE
3. **Test Item #68**: *"Not Happy. I recently purchased a used 8 GB View. It works well as a 8 GB View but will not support an 32 GB SD card."*
   - *True Label*: NEGATIVE
   - *Baseline*: POSITIVE (Dominant weights on "Happy", "works well")
   - *DistilBERT*: NEGATIVE
4. **Test Item #103**: *"NON-STOP ACTION... What's NOT to like about this movie? I'm not sure why this movie didn't do well. It has its faults but..."*
   - *True Label*: POSITIVE (Rhetorical question indicating love for film)
   - *Baseline*: NEGATIVE (Falsely penalized by capitalized "NOT" and "didn't do well")
   - *DistilBERT*: POSITIVE
5. **Test Item #134**: *"Not as good as 'What burns never returns'... but a very good pick anyway!"*
   - *True Label*: POSITIVE
   - *Baseline*: NEGATIVE (Negation prefix "Not as good" weighted higher than concluding clause)
   - *DistilBERT*: POSITIVE
6. **Integration #12 (Hindi)**: *"यह उत्पाद बिल्कुल भी काम नहीं करता, पैसे की पूरी बर्बादी।"*
   - *True Label*: NEGATIVE
   - *Baseline (Translated)*: POSITIVE (Saw "काम" [work] and "उत्पाद" without capturing "नहीं")
   - *DistilBERT*: NEGATIVE
7. **Test Item #48**: *"Well, it arrived on time and in OK condition, but after almost a month of taking it, I haven't noticed any change."*
   - *True Label*: NEGATIVE
   - *Baseline*: POSITIVE ("on time", "OK condition")
   - *DistilBERT*: NEGATIVE
8. **Test Item #76**: *"Fun movie, but the tape quality listed as 'NEW' has faded colors- VHS tapes just don't hold up."*
   - *True Label*: POSITIVE (Product catalog labeled movie positive, reviewer noted format limitation)
   - *Both Models*: NEGATIVE ("don't hold up", "faded")
9. **Test Item #130**: *"Perfect replacement for a lost charger. If you happen to misplace the factory charger, this is the exact replacement..."*
   - *True Label*: POSITIVE
   - *Both Models*: NEGATIVE (Words "lost", "misplace" dragged probability down)
10. **Test Item #136**: *"If you read the book, don't watch the movie! I really enjoyed this book, but I was very disappointed by the movie."*
    - *True Label*: NEGATIVE (Item is the DVD movie)
    - *Baseline*: POSITIVE (High frequency of "enjoyed this book")
    - *DistilBERT*: NEGATIVE

---

### Category B: Mixed Sentiment & Clause Conflicts (13 Cases)

11. **Test Item #8**: *"Smooth Feel but costly. I loved the product, but when I went back to get another bottle, I discovered it had nearly doubled in price."*
    - *True*: POSITIVE | *Baseline*: NEGATIVE | *DistilBERT*: NEGATIVE (Economic frustration overwhelmed sensory praise)
12. **Test Item #15**: *"Utopian fantasy. Hogan is a member of that group of sci-fi writers who present bold ideas but don't particularly write well."*
    - *True*: NEGATIVE | *Baseline*: POSITIVE ("bold ideas", "fantasy") | *DistilBERT*: NEGATIVE
13. **Test Item #18**: *"Cute but typical. So-so at its best. Great scenery. Good-looking men, but so typically chick-flick despite the true story."*
    - *True*: NEGATIVE | *Baseline*: POSITIVE | *DistilBERT*: POSITIVE (Superficial positive adjectives overpowered faint praise)
14. **Test Item #44**: *"How is this supposed to work? I got these for my jawbone headset. Many had said they work great, but for the life of me..."*
    - *True*: NEGATIVE | *Baseline*: POSITIVE ("work great") | *DistilBERT*: NEGATIVE
15. **Test Item #45**: *"Movie is awesome. Service comparable to Wal-Mart. I already know that I will love the movie. The problem is, shipping is awful."*
    - *True*: NEGATIVE | *Baseline*: NEGATIVE | *DistilBERT*: POSITIVE ("Movie is awesome... love the movie")
16. **Test Item #56**: *"Could have done without the 'laugh'. Very informative, content was well structured, though the instructor is a self-declared..."*
    - *True*: POSITIVE | *Baseline*: NEGATIVE | *DistilBERT*: NEGATIVE
17. **Test Item #115**: *"Fine for children, boring for adults. One has to be a young adult to enjoy this book. If you want to read about children..."*
    - *True*: NEGATIVE | *Both Models*: POSITIVE ("Fine", "enjoy")
18. **Test Item #153**: *"Not great, but not bad. Firstly, it is incredibly important that you DO NOT use the bundled software for this scanner."*
    - *True*: NEGATIVE | *Baseline*: NEGATIVE | *DistilBERT*: POSITIVE ("Not bad", "incredibly important")
19. **Test Item #2**: *"TWO DIFFERENT VERSIONS OF THE SAME SOUNDTRACK. Triplets of Belleville soundtrack and Belleville Rendezvous are the same disc..."*
    - *True*: POSITIVE | *Both Models*: NEGATIVE (Review is a public service advisory about duplicate purchase)
20. **Test Item #55**: *"Just Like New. Saved a lot of money. Looks brand spank'n new and works like a charm. I have purchased two other remanufactured..."*
    - *True*: POSITIVE | *Baseline*: NEGATIVE ("remanufactured") | *DistilBERT*: POSITIVE
21. **Integration Sample #14**: *"Battery backup is great, but screen cracked on day 2 and customer support refused replacement."*
    - *True*: NEGATIVE | *Baseline*: POSITIVE ("great") | *DistilBERT*: NEGATIVE
22. **Integration Sample #28 (Hindi)**: *"कैमरा ठीक-ठाक है लेकिन बैटरी बहुत जल्दी खत्म हो जाती है।"*
    - *True*: NEGATIVE | *Baseline*: POSITIVE | *DistilBERT*: NEGATIVE
23. **Test Item #138**: *"Canon NB-5L Battery for Canon camera. I purchased a Canon camera a couple of years ago and a Canon battery came with it..."*
    - *True*: POSITIVE | *Baseline*: NEGATIVE | *DistilBERT*: POSITIVE

---

### Category C: Sarcasm, Irony & Rhetorical Expressions (7 Cases)

24. **Hindi Test #26**: *"क्या शानदार फोन है, जेब में रखते ही हीटर बन जाता है।" (What a magnificent phone, turns into a heater in the pocket!)*
    - *True*: NEGATIVE | *Both Models*: POSITIVE (Fooled by "शानदार" [magnificent / fantastic])
25. **Hindi Test #27**: *"डिलीवरी इतनी तेज थी कि केवल दो महीने में आ गई!" (Delivery was so fast that it arrived in only two months!)*
    - *True*: NEGATIVE | *Both Models*: POSITIVE (Fooled by "तेज" [fast])
26. **Test Item #82**: *"I'm going nuts! PPLLEEAASSEE HURRY UP! I read this book when I was in grade 5, am now in grade 8 and have read it 5 times..."*
    - *True*: POSITIVE | *DistilBERT*: NEGATIVE ("going nuts", capitalized urgency parsed as distress)
27. **Test Item #83**: *"Nobody would believe this in a novel... only true crime!! It is hard to imagine how the three characters..."*
    - *True*: POSITIVE | *Baseline*: NEGATIVE ("hard to imagine", "Nobody would believe")
28. **Test Item #43**: *"RE: Wipeout originally by Surfaris. Google the instrumental Wipeout, and you will read the real story. I love the Surfaris..."*
    - *True*: NEGATIVE (Complaint about cover band misattribution) | *Both Models*: POSITIVE ("I love")
29. **Integration Sample #29 (Hindi)**: *"पैकेजिंग इतनी मजबूत थी कि खोलने के लिए हथौड़ा चाहिए था।" (Packaging was so strong that a hammer was required to open it.)*
    - *True*: NEGATIVE | *Both Models*: POSITIVE ("मजबूत" [strong])
30. **Integration Sample #7**: *"Oh wonderful, another software update that deleted all my custom shortcuts."*
    - *True*: NEGATIVE | *Baseline*: POSITIVE ("wonderful") | *DistilBERT*: NEGATIVE

---

### Category D: Text Length & Contextual Asymmetry (5 Cases)

31. **Test Item #47 (150 words)**: *"BALLISTIC BOMB. This tedious and unnecessary remake of the 1975 James Caan classic has little to recommend it. Director..."*
    - *True*: NEGATIVE | *Baseline*: POSITIVE (Long paragraph with classic film terminology dispersed TF-IDF weights)
32. **Test Item #62 (155 words)**: *"Alternate History meets Classic Horror. Kim Newman's gothic horror novel, 'Anno Dracula' is a fine riff of Bram Stoker's..."*
    - *True*: POSITIVE | *DistilBERT*: NEGATIVE (Horror-themed terms "vampire", "blood", "gothic" triggered negative prior)
33. **Test Item #74 (133 words)**: *"DELL E207WFP 20.1\" Digital/Analog TFT Flat Panel Monitor. Just a great product from Dell! I never regret purchasing this..."*
    - *True*: POSITIVE | *DistilBERT*: NEGATIVE ("never regret" double-negation in long text)
34. **Test Item #102 (32 words, terse)**: *"Damaged. Was starting to swell, which was why I was looking for another. No more purchases like this from me. Very disappointed."*
    - *True*: NEGATIVE | *DistilBERT*: POSITIVE (Short run-on sentence without punctuation confused subword tokenizer)
35. **Integration Sample #10 (Empty / Ultra-short)**: *"meh."*
    - *True*: AMBIGUOUS / NEUTRAL | *Models forced binary classification* -> Flagged for human review.

---

## 3. Concrete Engineering Mitigations Implemented in VaaniLens

1. **Confidence Threshold Gating (`CONF_THRESHOLD = 0.75`)**:
   - In ambiguous mixed-sentiment reviews (e.g. Test #8, #18, #153), model posterior confidence falls between 0.50 and 0.74.
   - The pipeline automatically triggers `needs_human_review = True`, preventing automated misclassification from driving high-stakes downstream workflows.
2. **LIME Interpretability Layer**:
   - For sarcasm and negation errors, LIME visualizes exact token weights. Human reviewers immediately observe if "शानदार" or "great" inappropriately dominated a negative review.
3. **Cross-Lingual Gender & Postposition Gating**:
   - Translation warnings alert operators whenever MarianMT introduces synthetic pronouns not present in the gender-neutral source text.
