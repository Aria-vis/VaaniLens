# Machine Translation Paradigm Showdown

A rigorous side-by-side comparison of Rule-Based Machine Translation (RBMT), Statistical Machine Translation (SMT), and Neural Machine Translation (NMT) for Hindi to English review text.

## Quantitative Performance

| Architecture | BLEU | chrF | Speed (s/sent) | Memory / Model Size | Key Limitation |
|---|---|---|---|---|---|
| **Rule-Based (RBMT)** | **1.88** | **22.65** | 0.0000s | < 1 MB | Severe vocabulary coverage gaps and rigid word order. |
| **Mini-SMT (IBM Model 1)** | **4.68** | **32.56** | 0.0001s | ~ 2 MB | Monotone word decoding, no target language model, disfluent English. |
| **Neural MT (MarianMT)** | **29.84** | **52.93** | 2.7944s | ~ 300 MB | High compute requirements, hallucinated gender pronouns. |

## Qualitative Analysis (10 Side-by-Side Test Examples)

| # | Hindi Source | Gold English Reference | Rule-based | Mini-SMT | Neural MT (MarianMT) |
|---|---|---|---|---|---|
| 1 | यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है। | This phone is very good, the battery lasts all day. | This is phone very good is battery all day runs. | This all very all all battery all all battery [है]. | That's very good call, the battery runs all day. |
| 2 | डिलीवरी बहुत देर से आई और सैमसंग का चार्जर खराब निकला। | The delivery was very late and the Samsung charger turned out to be faulty. | Delivery very late from came and charger of Samsung faulty turned out. | Turned very turned the turned and turned is charger not [निकला]. | The delivery came too late and the charge of Sam Youngs was bad. |
| 3 | बकवास प्रोडक्ट, पैसे बर्बाद। | Rubbish product, waste of money. | Rubbish [प्रोडक्ट] money [बर्बाद]. | Waste waste money. [बर्बाद]. | Fucking bullshit, money. |
| 4 | वह डॉक्टर है। | He is a doctor. | That is doctor. | Doctor. doctor. [है]. | She's a doctor. |
| 5 | कैमरे की क्वालिटी बहुत शानदार है। | The camera quality is superb. | [क्वालिटी] is of [कैमरे] very superb. | Camera the quality very superb. [है]. | The quality of the camera is amazing. |
| 6 | इस कीमत में यह सबसे बेहतरीन लैपटॉप है। | At this price, this is the best laptop. | [इस] is price in this [सबसे] excellent laptop. | Best best in this this best best [है]. | This cost it's the best flat. |
| 7 | हेडफ़ोन बहुत आरामदायक हैं और आवाज़ साफ़ है। | The headphones are very comfortable and the sound is clear. | Headphones is very [आरामदायक] are and sound [साफ़]. | Headphones very comfortable headphones and the headphones [है]. | Headphones are very comfortable and the voice is clear. |
| 8 | कपड़े की क्वालिटी बहुत बढ़िया है। | The quality of the fabric is very great. | [क्वालिटी] is of [कपड़े] very great. | Great. the quality very great. [है]. | The quality of the cloth is great. |
| 9 | यह चार्जर बहुत तेजी से चार्ज करता है। | This charger charges very rapidly. | This is charger very [तेजी] from [चार्ज] does. | This charger very charges the charges very [है]. | It's the charger charge too fast. |
| 10 | ग्राहक सेवा बहुत मददगार थी। | Customer service was very helpful. | Customer was service very [मददगार]. | Customer customer very customer [थी]. | The customer service was very helpful. |

## Why Each Approach Behaves As It Does

1. **Rule-Based Machine Translation**: Relies on a hand-crafted lexicon and grammar rules (e.g. postposition inversion and SOV->SVO movement). While computationally instantaneous, unmapped words produce bracketed '[OOV]' gaps, and complex sentence structures fail to translate fluently.
2. **Statistical Machine Translation (IBM Model 1)**: Estimates lexical translation probabilities $t(e|f)$ through Expectation-Maximization. While it captures probabilistic alignments without manual dictionaries, word-level monotone decoding generates ungrammatical target sentences without phrase coherence or syntax awareness.
3. **Neural Machine Translation (MarianMT)**: Utilizes a deep Sequence-to-Sequence Transformer architecture with multi-head self-attention. It produces fluent, grammatically sound English sentences by capturing long-range contextual dependencies across the entire sequence. However, it sometimes hallucinates gender pronouns on ungendered source inputs.
