# End-to-End Pipeline Integration Test & Failure Analysis Log (GATE-2)

Evaluation of 30 challenging multilingual review samples across English and Hindi, covering edge cases: negation, sarcasm, mixed sentiment, empty inputs, extreme length, and gender pronoun inference.

| ID | Lang | Category | Input Preview | Sentiment (Score) | Review Flag | Warnings | Latency |
|---|---|---|---|---|---|---|---|
| 1 | EN | positive | This laptop is blazingly fast and the OL... | POSITIVE (1.00) | ✅ OK | None | 31536ms |
| 2 | EN | negative | The charger caught fire on the second da... | NEGATIVE (1.00) | ✅ OK | None | 67ms |
| 3 | EN | negation | This budget headset is not bad at all fo... | POSITIVE (1.00) | ✅ OK | None | 52ms |
| 4 | EN | sarcasm | Oh fantastic, another software update th... | POSITIVE (1.00) | ✅ OK | None | 62ms |
| 5 | EN | mixed | The Redmi Note arrived in Patna two days... | POSITIVE (1.00) | ✅ OK | None | 82ms |
| 6 | EN | empty |  | POSITIVE (0.50) | ⚠️ Flagged | Input is empty. | 0ms |
| 7 | EN | long | I have been testing this smartphone exte... | POSITIVE (0.98) | ✅ OK | None | 692ms |
| 8 | EN | entities | Purchased Samsung Galaxy S24 from Amazon... | POSITIVE (0.88) | ✅ OK | None | 91ms |
| 9 | EN | neutral | The package arrived on Tuesday in a brow... | NEGATIVE (1.00) | ✅ OK | None | 77ms |
| 10 | EN | short_positive | Superb! | POSITIVE (1.00) | ✅ OK | None | 45ms |
| 11 | EN | short_negative | Horrible waste of money. | NEGATIVE (1.00) | ✅ OK | None | 73ms |
| 12 | EN | identity_term | I am a Hindu and I found this religious ... | POSITIVE (1.00) | ✅ OK | None | 107ms |
| 13 | EN | identity_term | As a Christian, I think this holiday dec... | POSITIVE (1.00) | ✅ OK | None | 84ms |
| 14 | EN | typo | Gud produckt, werks fine bt dilvery slow... | NEGATIVE (1.00) | ✅ OK | None | 74ms |
| 15 | EN | special_chars | Great product!!! 10/10 **** Loved it #aw... | POSITIVE (1.00) | ✅ OK | None | 70ms |
| 16 | HI | positive | यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चल... | NEGATIVE (0.61) | ⚠️ Flagged | Low confidence sentiment (0.61 < 0.75). Needs human review. | 5523ms |
| 17 | HI | negative | डिलीवरी बहुत देर से आई और सैमसंग का चार्... | NEGATIVE (1.00) | ✅ OK | None | 1090ms |
| 18 | HI | short_negative | बकवास प्रोडक्ट, पैसे बर्बाद। | NEGATIVE (1.00) | ✅ OK | None | 433ms |
| 19 | HI | gender_pronoun | वह डॉक्टर है और अस्पताल में बहुत मेहनत क... | NEGATIVE (0.91) | ✅ OK | Pronoun gender was inferred by the translation model, not stated in the source. | 795ms |
| 20 | HI | negation | यह बिल्कुल भी खराब नहीं है, बल्कि बहुत ब... | POSITIVE (1.00) | ✅ OK | None | 692ms |
| 21 | HI | sarcasm | अरे वाह, एक और प्रोडक्ट जो दो दिन में बं... | NEGATIVE (1.00) | ✅ OK | None | 1930ms |
| 22 | HI | mixed | दिखने में तो अच्छा है, लेकिन काम बिल्कुल... | NEGATIVE (1.00) | ✅ OK | None | 1523ms |
| 23 | HI | entities | मैंने पटना के बाजार से यह सोनी का हेडफोन... | NEGATIVE (0.95) | ✅ OK | None | 808ms |
| 24 | HI | empty |     | POSITIVE (0.50) | ⚠️ Flagged | Input is empty. | 0ms |
| 25 | HI | long | यह स्मार्टफोन पिछले एक महीने से उपयोग कर... | POSITIVE (1.00) | ✅ OK | None | 3294ms |
| 26 | HI | identity_term | मैं एक भारतीय हूं और इस स्वदेशी ब्रांड क... | POSITIVE (1.00) | ✅ OK | None | 948ms |
| 27 | HI | short_positive | शानदार! | POSITIVE (1.00) | ✅ OK | None | 220ms |
| 28 | HI | service | ग्राहक सेवा ने तुरंत फोन उठाया और नया चा... | NEGATIVE (0.87) | ✅ OK | None | 657ms |
| 29 | HI | hinglish_slang | ये फोन एकदम झकास है, पर इसका चार्जर बकवा... | NEGATIVE (0.99) | ✅ OK | None | 2499ms |
| 30 | HI | complex | सोचा था बहुत बढ़िया होगा, पर यह तो महा ब... | NEGATIVE (1.00) | ✅ OK | None | 1441ms |

## Edge Case Findings & Qualitative Observations

- **Sample #4 (Sarcasm in EN)**: Surface lexical polarity is highly positive ('wonderful', 'बहुत खूब'), leading DistilBERT to predict POSITIVE with high confidence. Sarcasm remains a known limitation of non-fine-tuned transformers.

## Summary Statistics
- **Total Samples Processed**: 30
- **Exceptions / Crashes**: 0
- **Human Review Required Flags Triggered**: 3 / 30 (10.0%)
- **Pass Status**: PASSED (GATE-2 Cleared)
