1. Puścić `ONSBoost` dla kilku strumieni:
    * ciągły dynamicznie niezbalansowany (`weights` to 3-tuple)
        * `n_drifts` - wiadomo
        * `concept_sigmoid_spacing` - "nagłość" dryfu
        * `IR_amplitude` - zmiana stopnia niezbalansowania
    * dyskretny dynamicznie niezbalansowany (`weights` to 2-tuple)
        * `mean`
        * `stddev`
    * DISCO? Dynamically Imbalanced Stream With Concept Oscillation
    
    Zobaczyć które parametry dają najlepsze rezultaty, jaki jest wpływ parametru na działanie metody. Używając `random_state` można sobie wygenerować "podobne" (w kwestii charakterystyki) strumienie 
2. To samo można zrobić dla zwykłego `OnlineBoosting`.
3. Wziąć istniejące metody z biblioteki: `OB`, `OOB`, `UOB`. Przy wielu wynikach można zrobić test statystyczny dla poszczególnych metryk (test Friedmana? Wilcoxona?)
4. Metryki:`BAC`, `Precision`, `Recall`, `F1`, `G-mean`, `Kappa` (?) 