# Sentiment Model Demographic Bias Audit Summary

| Model | Demographic Category | Max Mean P(Pos) | Min Mean P(Pos) | Disparity | Flag (> 0.05) |
|---|---|---|---|---|---|
| baseline | Religion | 0.5067 (an atheist) | 0.4412 (a Muslim) | 0.0655 | ⚠️ YES |
| baseline | Gender | 0.4679 (a man) | 0.4546 (a non-binary person) | 0.0133 | ✅ NO |
| baseline | Region | 0.4760 (a Tamil speaker) | 0.4529 (a Bihari) | 0.0231 | ✅ NO |
| baseline | Nationality | 0.4999 (an American) | 0.4529 (a Pakistani) | 0.0470 | ✅ NO |
| transformer | Religion | 0.6238 (a Christian) | 0.5109 (an atheist) | 0.1129 | ⚠️ YES |
| transformer | Gender | 0.6228 (a man) | 0.5172 (a non-binary person) | 0.1055 | ⚠️ YES |
| transformer | Region | 0.6166 (a Bengali) | 0.5606 (a Bihari) | 0.0559 | ⚠️ YES |
| transformer | Nationality | 0.6231 (an Indian) | 0.5958 (a Pakistani) | 0.0273 | ✅ NO |

## Plain-Language Interpretation

Disparities exceeding the 0.05 fairness threshold were detected:

- **baseline on religion**: Disparity of 0.0655 between 'an atheist' (0.5067) and 'a Muslim' (0.4412).
- **transformer on religion**: Disparity of 0.1129 between 'a Christian' (0.6238) and 'an atheist' (0.5109).
- **transformer on gender**: Disparity of 0.1055 between 'a man' (0.6228) and 'a non-binary person' (0.5172).
- **transformer on region**: Disparity of 0.0559 between 'a Bengali' (0.6166) and 'a Bihari' (0.5606).

**Ethical Implication**: Pretrained language models retain social biases from large uncurated internet pretraining corpora. Adding identity tokens changes the model's baseline optimism even when the surrounding review sentiment is held identical.

## Implemented Mitigations
1. **Human Review Gate**: Reviews with sentiment confidence < 0.75 are flagged for human oversight.
2. **LIME Interpretability**: Highlights word-level attributions so auditors can see if demographic tokens improperly influenced the decision.
