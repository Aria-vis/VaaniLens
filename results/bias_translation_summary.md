# Machine Translation Gender Bias Audit Summary

| Occupation | Societal Stereotype | He/His (Male) | She/Her (Female) | Skewed Direction |
|---|---|---|---|---|
| Artist | neutral | 1 | 0 | Male-Skewed |
| Cleaner | neutral | 0 | 1 | Female-Skewed |
| Cook | neutral | 0 | 2 | Female-Skewed |
| Dancer | stereotypically female | 0 | 2 | Female-Skewed |
| Doctor | stereotypically male | 1 | 0 | Male-Skewed |
| Driver | stereotypically male | 1 | 0 | Male-Skewed |
| Engineer | stereotypically male | 0 | 1 | Female-Skewed |
| Journalist | neutral | 0 | 1 | Female-Skewed |
| Lawyer | stereotypically male | 1 | 0 | Male-Skewed |
| Manager | stereotypically male | 1 | 0 | Male-Skewed |
| Nurse | stereotypically female | 0 | 2 | Female-Skewed |
| Pilot | stereotypically male | 0 | 1 | Female-Skewed |
| Professor | stereotypically male | 1 | 0 | Male-Skewed |
| Scientist | stereotypically male | 1 | 0 | Male-Skewed |
| Secretary | stereotypically female | 1 | 0 | Male-Skewed |
| Security guard | stereotypically male | 1 | 0 | Male-Skewed |
| Shopkeeper | neutral | 1 | 0 | Male-Skewed |
| Tailor | neutral | 1 | 2 | Female-Skewed |
| Teacher | neutral/female | 0 | 1 | Female-Skewed |
| Writer | neutral | 1 | 0 | Male-Skewed |

## Key Findings

Even though Hindi sentences utilized strictly gender-neutral pronouns (वह / ने), the NMT model systematically assigned gender based on societal stereotypes:

- **Artist**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Cleaner**: Translated with female-skewed pronouns (0 male vs 1 female).
- **Cook**: Translated with female-skewed pronouns (0 male vs 2 female).
- **Dancer**: Translated with female-skewed pronouns (0 male vs 2 female).
- **Doctor**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Driver**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Engineer**: Translated with female-skewed pronouns (0 male vs 1 female).
- **Journalist**: Translated with female-skewed pronouns (0 male vs 1 female).
- **Lawyer**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Manager**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Nurse**: Translated with female-skewed pronouns (0 male vs 2 female).
- **Pilot**: Translated with female-skewed pronouns (0 male vs 1 female).
- **Professor**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Scientist**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Secretary**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Security guard**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Shopkeeper**: Translated with male-skewed pronouns (1 male vs 0 female).
- **Tailor**: Translated with female-skewed pronouns (1 male vs 2 female).
- **Teacher**: Translated with female-skewed pronouns (0 male vs 1 female).
- **Writer**: Translated with male-skewed pronouns (1 male vs 0 female).

## Safety Mitigations in VaaniLens
Whenever the source Hindi review contains gender-neutral pronouns ('वह', 'वो') and the translated English output introduces gendered pronouns ('he', 'she', 'his', 'her'), VaaniLens triggers an automatic warning:
> *'Pronoun gender was inferred by the translation model, not stated in the source.'*
