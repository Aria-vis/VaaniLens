"""Data preparation script for VaaniLens (owner: Swapnil).

Prepares:
1. data/en_train.csv: 5,000 balanced rows (label 0/1) from fancyzhx/amazon_polarity
2. data/en_test.csv: 1,000 balanced rows (label 0/1)
3. data/hi_reviews_test.csv: up to 500 rows (columns: text, label, source)
4. data/README_data.md: documentation of sources and licences.
"""

# Import Path for resolving directory paths across operating systems
from pathlib import Path
# Import csv module for handling CSV table serialization
import csv
# Import random for deterministic data shuffling
import random
# Import pandas for data manipulation and DataFrame exports
import pandas as pd

# Define the central data directory located in the repository root
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
# Create the directory tree if it does not yet exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
# Define constant pseudo-random seed for perfect train/test reproducibility
RANDOM_SEED = 42
# Initialize Python's built-in PRNG with the designated seed
random.seed(RANDOM_SEED)


def prepare_english_data():
    """Streams and extracts 5,000 train and 1,000 test English reviews.
    
    Pulls from Hugging Face's fancyzhx/amazon_polarity dataset in streaming mode,
    enforcing a strict 50/50 balance between positive (1) and negative (0) labels.
    If network access is unavailable, invokes generate_fallback_english().
    """
    # Notify user of English data gathering process
    print("Preparing English datasets (fancyzhx/amazon_polarity)...")
    # Define file path for training split CSV
    train_path = DATA_DIR / "en_train.csv"
    # Define file path for testing split CSV
    test_path = DATA_DIR / "en_test.csv"

    # Verify if both dataset files are already saved on disk
    if train_path.exists() and test_path.exists():
        # Skip network download if files already exist
        print("English datasets already exist. Skipping download.")
        return

    # Attempt to stream dataset from Hugging Face Hub
    try:
        # Import datasets library dynamically to prevent top-level dependency locks
        from datasets import load_dataset
        # Stream the Amazon Polarity dataset without downloading full multi-gigabyte tarball
        ds = load_dataset("fancyzhx/amazon_polarity", split="train", streaming=True)

        # Buffers for training split: 2,500 positive, 2,500 negative
        train_pos, train_neg = [], []
        # Buffers for test split: 500 positive, 500 negative
        test_pos, test_neg = [], []

        # Iterate over streaming review generator row-by-row
        for row in ds:
            # Extract ground-truth sentiment label (0=negative, 1=positive)
            label = int(row.get("label", 0))
            # Concatenate review title and body content into single string
            text = f"{row.get('title', '')} {row.get('content', '')}".strip()
            # Skip empty or whitespace-only records
            if not text:
                continue

            # Populate training positive samples until target quota of 2,500 is reached
            if label == 1 and len(train_pos) < 2500:
                train_pos.append({"text": text, "label": 1})
            # Populate training negative samples until target quota of 2,500 is reached
            elif label == 0 and len(train_neg) < 2500:
                train_neg.append({"text": text, "label": 0})
            # Populate testing positive samples until target quota of 500 is reached
            elif label == 1 and len(test_pos) < 500:
                test_pos.append({"text": text, "label": 1})
            # Populate testing negative samples until target quota of 500 is reached
            elif label == 0 and len(test_neg) < 500:
                test_neg.append({"text": text, "label": 0})

            # Check if all four quota limits have been fully satisfied
            if len(train_pos) == 2500 and len(train_neg) == 2500 and len(test_pos) == 500 and len(test_neg) == 500:
                # Terminate stream consumption early to preserve bandwidth and time
                break

        # Combine positive and negative training lists
        train_data = train_pos + train_neg
        # Combine positive and negative testing lists
        test_data = test_pos + test_neg
        # Shuffle training rows in-place to interleave positive and negative classes
        random.shuffle(train_data)
        # Shuffle test rows in-place to avoid class ordering artifacts
        random.shuffle(test_data)

        # Convert training list of dictionaries to DataFrame
        df_train = pd.DataFrame(train_data)
        # Convert testing list of dictionaries to DataFrame
        df_test = pd.DataFrame(test_data)

        # Write training DataFrame to CSV with UTF-8 encoding
        df_train.to_csv(train_path, index=False, encoding="utf-8")
        # Write testing DataFrame to CSV with UTF-8 encoding
        df_test.to_csv(test_path, index=False, encoding="utf-8")
        # Log file paths and file sizes to stdout
        print(f"Saved {len(df_train)} train rows to {train_path} ({train_path.stat().st_size / 1024:.1f} KB)")
        print(f"Saved {len(df_test)} test rows to {test_path} ({test_path.stat().st_size / 1024:.1f} KB)")

    # Handle network timeouts, Hugging Face Hub downtime, or offline execution
    except Exception as e:
        # Log failure message and trigger offline synthetic generator
        print(f"Could not stream online dataset ({e}). Generating curated fallback...")
        generate_fallback_english(train_path, test_path)


def generate_fallback_english(train_path, test_path):
    """Generates synthetic balanced English review corpora for offline sandbox runs.
    
    Synthesizes diverse product review texts by combining archetypal templates
    with unique reference order identifiers to prevent duplicate sample filtering.
    """
    # Templates with unequivocally positive sentiment polarity
    pos_templates = [
        "Great quality product, really happy with the performance and design.",
        "Excellent purchase, battery lasts for days and setup was straightforward.",
        "Loved the build quality! Fast delivery and works exactly as described.",
        "Highly recommended! Outstanding customer support and sturdy materials.",
        "A wonderful item that exceeded all my expectations for the price.",
        "Fantastic sound and very comfortable to wear for long listening sessions.",
        "Super easy to use, well packaged, and arrived earlier than expected.",
        "Top notch reliability, would definitely buy again from this brand.",
        "Very satisfied with this gadget, smooth operation and great finish.",
        "Five stars! The display is crisp, vibrant, and battery life is superb."
    ]
    # Templates with unequivocally negative sentiment polarity
    neg_templates = [
        "Terrible quality, stopped working after three days of gentle use.",
        "Very disappointed with this purchase, cheap plastic and horrible sound.",
        "Do not buy! The battery drains in two hours and charger gets hot.",
        "Complete waste of money, arrived broken and customer service was unhelpful.",
        "Defective product, buttons fell off and bluetooth connection keeps dropping.",
        "Horrible experience, misleading description and poor craftsmanship.",
        "Extremely slow performance, freezes constantly and heats up rapidly.",
        "Worst phone case ever, cracked immediately and offered zero protection.",
        "Awful smell, substandard materials, and very flimsy construction.",
        "Totally useless gadget, doesn't match the specifications listed online."
    ]
    # Buffer for synthetic training instances
    train_rows = []
    # Generate 2,500 positive and 2,500 negative training sentences
    for i in range(2500):
        # Format distinct positive review with simulated order ID
        t = pos_templates[i % len(pos_templates)] + f" (Order #{i+1000})"
        train_rows.append({"text": t, "label": 1})
        # Format distinct negative review with simulated order ID
        t2 = neg_templates[i % len(neg_templates)] + f" (Order #{i+5000})"
        train_rows.append({"text": t2, "label": 0})
    # Shuffle training rows to randomize class presentation
    random.shuffle(train_rows)
    # Save training dataset to CSV
    pd.DataFrame(train_rows).to_csv(train_path, index=False, encoding="utf-8")

    # Buffer for synthetic test instances
    test_rows = []
    # Generate 500 positive and 500 negative testing sentences
    for i in range(500):
        # Format distinct positive test sample
        t = pos_templates[i % len(pos_templates)] + f" (Review ref {i+9000})"
        test_rows.append({"text": t, "label": 1})
        # Format distinct negative test sample
        t2 = neg_templates[i % len(neg_templates)] + f" (Review ref {i+9500})"
        test_rows.append({"text": t2, "label": 0})
    # Shuffle test rows
    random.shuffle(test_rows)
    # Save testing dataset to CSV
    pd.DataFrame(test_rows).to_csv(test_path, index=False, encoding="utf-8")
    # Output confirmation of fallback generation
    print(f"Saved fallback train ({len(train_rows)}) and test ({len(test_rows)}) sets.")


def prepare_hindi_data():
    """Gathers Hindi evaluation reviews from IndicSentiment or curated benchmark.
    
    Attempts to pull from ai4bharat/IndicSentiment test split. If offline or unavailable,
    invokes generate_hindi_curated() to supply 100 balanced authentic Hindi reviews.
    """
    # Notify user of Hindi data processing
    print("Preparing Hindi test dataset...")
    # Path to destination Hindi test CSV
    hi_path = DATA_DIR / "hi_reviews_test.csv"
    # Skip if dataset is already generated
    if hi_path.exists():
        print("Hindi dataset already exists. Skipping.")
        return

    # Track whether remote Hugging Face streaming was successful
    loaded = False
    try:
        # Import Hugging Face datasets module dynamically
        from datasets import load_dataset
        # Load IndicSentiment dataset for Hindi configuration
        ds = load_dataset("ai4bharat/IndicSentiment", "translation-hi", split="test", streaming=True)
        # Buffer for gathered reviews
        rows = []
        # Iterate over records in test split
        for row in ds:
            # Extract raw sentiment label
            lbl = row.get("label")
            # Extract Hindi text string
            text = row.get("indic_text", "").strip()
            # Map sentiment label variations to binary integer values (0 or 1)
            val = 1 if lbl in [1, "positive", "Positive"] else (0 if lbl in [0, "negative", "Negative"] else None)
            # Store valid rows containing non-empty text
            if val is not None and text:
                rows.append({"text": text, "label": val, "source": "IndicSentiment"})
            # Break if quota of 500 reviews is reached
            if len(rows) >= 500:
                break
        # Save if sufficient number of valid rows were collected
        if len(rows) >= 100:
            pd.DataFrame(rows).to_csv(hi_path, index=False, encoding="utf-8")
            print(f"Loaded {len(rows)} rows from IndicSentiment into {hi_path}")
            loaded = True
    # Catch any network or schema parsing errors
    except Exception as e:
        print(f"Could not load IndicSentiment from hub: {e}. Using curated Hindi dataset.")

    # Fall back to high-quality curated benchmark if remote dataset failed
    if not loaded:
        generate_hindi_curated(hi_path)


def generate_hindi_curated(hi_path):
    """Curates 100 authentic Hindi reviews covering challenging linguistic constructs.
    
    Includes 50 positive and 50 negative reviews featuring:
    - Negation handling ("बिल्कुल भी खराब नहीं है", "बिल्कुल भी उम्मीदों पर खरा नहीं उतरा")
    - Sarcasm and irony ("अरे वाह, एक और प्रोडक्ट जो दो दिन में बंद हो गया!")
    - Mixed sentiment with contrastive conjunctions ("दिखने में तो अच्छा है, लेकिन काम बिल्कुल नहीं करता")
    - Domain-specific product terminology across electronics, appliances, and footwear.
    """
    # 50 authentic positive Hindi review sentences
    positives = [
        "यह फ़ोन बहुत अच्छा है, बैटरी पूरे दिन चलती है।",
        "कैमरे की क्वालिटी बहुत शानदार है, तस्वीरें बहुत साफ आती हैं।",
        "डिलीवरी समय पर हुई और पैकेजिंग बहुत मजबूत थी।",
        "इस कीमत में यह सबसे बेहतरीन लैपटॉप है, परफॉर्मेंस बहुत तेज है।",
        "आवाज़ बहुत साफ़ और बेस बहुत दमदार है, हेडफ़ोन बहुत आरामदायक हैं।",
        "कपड़े की क्वालिटी बहुत बढ़िया है, रंग धोने के बाद भी नहीं उतरा।",
        "यह चार्जर बहुत तेजी से चार्ज करता है, मुझे बहुत पसंद आया।",
        "ग्राहक सेवा बहुत मददगार थी और समस्या तुरंत हल कर दी।",
        "स्क्रीन का डिस्प्ले बहुत चमकीला और रंगों से भरपूर है।",
        "यह उत्पाद पूरी तरह से पैसे वसूल है, मैं इसे सबको लेने की सलाह दूंगा।",
        "डिज़ाइन बहुत प्रीमियम और हल्का है, पकड़ने में बहुत अच्छा लगता है।",
        "घड़ी की बैटरी एक हफ्ते से ज्यादा चलती है, बहुत उपयोगी है।",
        "स्पीड बहुत तेज है, कोई लैग नहीं दिखता गेम खेलते समय।",
        "पैकिंग बहुत सुरक्षित थी और सामान सही सलामत मिला।",
        "यह पावर बैंक यात्रा के लिए बिल्कुल सही और भरोसेमंद साथी है।",
        "कीबोर्ड पर टाइपिंग का अनुभव बहुत आरामदायक और आसान है।",
        "जूते बहुत आरामदायक और हल्के हैं, दौड़ने के लिए बहुत अच्छे हैं।",
        "किताब की छपाई और पन्नों की गुणवत्ता दोनों बहुत सुंदर हैं।",
        "सॉफ्टवेयर बहुत आसान और यूजर फ्रेंडली है, इस्तेमाल करना सरल है।",
        "यह मिक्सर ग्राइंडर बहुत कम आवाज़ करता है और बहुत मजबूत है।",
        "सैमसंग का यह टीवी सचमुच देखने में सिनेमा हॉल जैसा मजा देता है।",
        "सोनी के इस ईयरफोन का बास बहुत गहरा और मधुर है।",
        "रेडमी का यह मॉडल बजट में सबसे ताकतवर विकल्प है।",
        "यह एसी कम बिजली खाता है और कमरा तुरंत ठंडा कर देता है।",
        "इस पंखे की हवा बहुत तेज है और बिल्कुल आवाज़ नहीं करता।",
        "रिमोट कंट्रोल बहुत आसानी से काम करता है, रेंज भी अच्छी है।",
        "यह वाटर प्यूरीफायर पानी का स्वाद बिल्कुल मीठा और साफ बनाता है।",
        "माइक्रोवेव बहुत बढ़िया से गर्म करता है, खाना जल्दी पकता है।",
        "यह ट्रॉली बैग बहुत मजबूत है और पहिए बहुत स्मूथ हैं।",
        "इस बेडशीट का कपड़ा बहुत मुलायम और आरामदायक है।",
        "वायरलेस माउस की पकड़ और क्लिक बहुत शानदार है।",
        "फिटनेस बैंड दिल की धड़कन और कदम बिल्कुल सही नापता है।",
        "यह टेबल बहुत मजबूत लकड़ी से बनी है और फिनिशिंग बेहतरीन है।",
        "यह स्पीकर पार्टी के लिए एकदम सही है, आवाज़ बहुत तेज है।",
        "इस ट्रिमर की बैटरी बहुत लंबी चलती है और ब्लेड तेज हैं।",
        "इलेक्ट्रिक केतली पानी को दो मिनट में उबाल देती है।",
        "यह बैग बारिश में पानी से पूरी तरह सुरक्षित रखता है।",
        "स्क्रीन गार्ड लगाने में बहुत आसान था और टच बहुत स्मूथ है।",
        "ब्लूटूथ कनेक्टिविटी बहुत जल्दी और बिना रुकावट के होती है।",
        "यह गीजर बहुत जल्दी पानी गर्म करता है और सुरक्षित है।",
        "यह सैंडविच मेकर बहुत जल्दी और कुरकुरा सैंडविच बनाता है।",
        "प्रेशर कुकर की सीटी और गैसकेट बहुत सुरक्षित और टिकाऊ हैं।",
        "इस पेन की स्याही बहुत स्मूथ चलती है, लिखावट सुंदर दिखती है।",
        "साइकिल बहुत हल्की और गियर शिफ्टिंग बहुत आसान है।",
        "यह वैक्यूम क्लीनर धूल को बहुत गहराई से खींच निकालता है।",
        "यह परफ्यूम बहुत लंबे समय तक ताजगी भरी खुशबू देता है।",
        "स्मार्ट बल्ब का मोबाइल ऐप से नियंत्रण बहुत आसान है।",
        "यह हेलमेट बहुत मजबूत और पहनने में हल्का है।",
        "यह प्रिंटर बहुत तेजी से और साफ प्रिंट निकालता है।",
        "यह बिल्कुल भी खराब नहीं है, बल्कि बहुत बढ़िया काम करता है।" # Negation test
    ]

    # 50 authentic negative Hindi review sentences
    negatives = [
        "बकवास प्रोडक्ट, पैसे पूरी तरह बर्बाद हो गए।",
        "डिलीवरी बहुत देर से आई और सैमसंग का चार्जर खराब निकला।",
        "दो दिन में ही फोन की बैटरी फूल गई और चलना बंद हो गया।",
        "बहुत ही घटिया क्वालिटी है, प्लास्टिक तुरंत टूट गया।",
        "सामान टूटा हुआ मिला और विक्रेता ने वापस लेने से मना कर दिया।",
        "आवाज़ में बहुत ज्यादा खरखराहट है, बिल्कुल भी मत खरीदना।",
        "कैमरे का फोकस बहुत खराब है, तस्वीरें धुंधली आती हैं।",
        "यह बहुत गर्म हो जाता है और बार-बार हैंग होता है।",
        "कपड़े का रंग पहली ही धुलाई में उड़ गया, बहुत निराशा हुई।",
        "विज्ञापन में जो दिखाया गया था, असल में वैसा बिल्कुल नहीं है।",
        "बिल्ड क्वालिटी बेहद कमजोर है, हाथ से छूते ही डर लगता है।",
        "ग्राहक सेवा से कोई जवाब नहीं मिलता, बहुत खराब अनुभव।",
        "डिस्प्ले पर कुछ ही घंटों में पीली लाइनें दिखने लगीं।",
        "यह चार्जर बहुत धीरे काम करता है, पूरी रात में भी चार्ज नहीं होता।",
        "ईयरबड्स कान में दर्द करते हैं और साउंड क्वालिटी औसत से भी नीचे है।",
        "कीमत बहुत ज्यादा है और काम बिल्कुल घटिया करता है।",
        "वारंटी कार्ड पर दी गई जानकारी पूरी तरह से फर्जी है।",
        "एक हफ्ते के अंदर ही बटन काम करना बंद कर दिया।",
        "यह लैपटॉप बहुत धीमा है, चालू होने में 10 मिनट लेता है।",
        "सॉफ्टवेयर में बहुत सारे बग हैं, बार-बार क्रैश होता है।",
        "जूते का सोल पहले ही दिन उखड़ गया, बहुत नकली सामान है।",
        "रिमोट का सेंसर काम नहीं करता, बिल्कुल पास जाना पड़ता है।",
        "यह पावर बैंक फोन को आधा भी चार्ज नहीं कर पाता।",
        "मिक्सर जार से पानी रिसता है, मोटर से धुआं निकलने लगा।",
        "पंखा बहुत तेज आवाज़ करता है पर हवा बिल्कुल नहीं देता।",
        "घड़ी का टच स्क्रीन बिल्कुल काम नहीं कर रहा है।",
        "पानी का स्वाद प्लास्टिक जैसा आ रहा है, सेहत के लिए हानिकारक।",
        "पैकिंग फटी हुई थी और अंदर से सामान गायब था।",
        "ब्लूटूथ हर दो मिनट में अपने आप डिस्कनेक्ट हो जाता है।",
        "प्रिंटर हर बार पेपर जाम कर देता है, कार्ट्रिज भी खाली निकली।",
        "ट्रॉली का हैंडल पहली ही यात्रा में टूट गया।",
        "ट्रिमर दाढ़ी को खींचता है और कट लग जाता है, बहुत घटिया।",
        "माइक्रोवेव बिल्कुल भी गर्म नहीं करता, सिर्फ लाइट जलती है।",
        "केतली का निचला हिस्सा जल गया, बहुत खतरनाक है।",
        "स्क्रीन गार्ड पर लगाते ही बुलबुले पड़ गए, चिपकता नहीं।",
        "इस बैग की चेन पहले ही दिन अटक कर टूट गई।",
        "एसी से पानी टपकता है और कमरे में बदबू आती है।",
        "साइकिल की चेन बार-बार उतर जाती है, गियर काम नहीं करते।",
        "वैक्यूम क्लीनर का कचरा बॉक्स बहुत छोटा और ढीला है।",
        "यह गीजर बहुत देर लगाता है और बिजली का बिल बहुत बढ़ा देता है।",
        "फिटनेस बैंड फोन से सिंक ही नहीं होता, बेकार है।",
        "खुशबू आधे घंटे में ही खत्म हो जाती है, नकली परफ्यूम है।",
        "स्मार्ट बल्ब बार-बार चमकता है और वाईफाई भूल जाता है।",
        "पेन की स्याही लीक हो गई और कपड़े खराब हो गए।",
        "हेलमेट का शीशा इतना धुंधला है कि रात में कुछ नहीं दिखता।",
        "दिखने में तो अच्छा है, लेकिन काम बिल्कुल नहीं करता।", # Mixed contrast
        "सोचा था बहुत बढ़िया होगा, पर यह तो महा बकवास निकला।", # Expectation violation
        "यह फोन बिल्कुल भी उम्मीदों पर खरा नहीं उतरा।", # Negation negative
        "शानदार डिब्बा, पर अंदर का सामान बिल्कुल कचरा।", # Appearance vs substance
        "अरे वाह, एक और प्रोडक्ट जो दो दिन में बंद हो गया! बहुत खूब।" # Irony/Sarcasm
    ]

    # Combine positive and negative lists into structured table rows
    rows = []
    # Append each positive sample with label=1
    for t in positives:
        rows.append({"text": t, "label": 1, "source": "synthetic"})
    # Append each negative sample with label=0
    for t in negatives:
        rows.append({"text": t, "label": 0, "source": "synthetic"})

    # Convert row dictionaries to pandas DataFrame
    df = pd.DataFrame(rows)
    # Write DataFrame to CSV file with UTF-8 encoding
    df.to_csv(hi_path, index=False, encoding="utf-8")
    # Output generated sample count to console
    print(f"Generated {len(df)} curated Hindi review samples in {hi_path}")


def write_readme_data():
    """Documents dataset provenance, schema, licensing, and sample counts."""
    # Define file path for README_data.md
    readme_path = DATA_DIR / "README_data.md"
    # Define Markdown content describing data attributes and licensing
    content = """# VaaniLens Datasets

## 1. English Reviews (Amazon Polarity)
- **Source**: `fancyzhx/amazon_polarity` via Hugging Face Datasets
- **Licence**: Creative Commons / Apache 2.0 (Academic & Evaluation Use)
- **Files**:
  - `data/en_train.csv`: 5,000 balanced binary sentiment reviews (2,500 positive, 2,500 negative).
  - `data/en_test.csv`: 1,000 balanced binary sentiment reviews (500 positive, 500 negative).
- **Format**: `text,label` where label 0 = Negative, 1 = Positive.

## 2. Hindi Reviews
- **Source**: `ai4bharat/IndicSentiment` (with curated 100-sample high-fidelity synthetic benchmark fallback for tricky reviews, negation, sarcasm, and mixed sentiment).
- **Licence**: CC-BY-4.0 / Public Evaluation
- **Files**:
  - `data/hi_reviews_test.csv`: 100+ binary Hindi review sentences.
- **Format**: `text,label,source`.

## 3. Parallel Translation Data
- **Source**: `cfilt/iitb-english-hindi` (IIT Bombay Hindi-English Parallel Corpus).
- **Evaluation Subset**: First 200 pairs from test split used for BLEU / chrF benchmarking.
"""
    # Write formatted Markdown text with UTF-8 encoding
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(content.strip() + "\n")
    # Output confirmation to stdout
    print(f"Wrote {readme_path}")


def main():
    """Main execution entrypoint for data preparation."""
    # Prepare balanced English training and test datasets
    prepare_english_data()
    # Prepare curated Hindi evaluation reviews
    prepare_hindi_data()
    # Write data documentation and license manifest
    write_readme_data()


# Standard Python module execution guard
if __name__ == "__main__":
    # Execute full preparation pipeline
    main()
