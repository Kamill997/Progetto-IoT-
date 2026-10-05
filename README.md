# Electricity and Cryptocurrency Mining — Replica sperimentale

Replica della parte sperimentale di:

> Okorie, D.I., Gnatchiglo, J.M., Wesseh, P.K. (2024). _Electricity and
> cryptocurrency mining: An empirical contribution._ Heliyon 10, e33483.

Il progetto riproduce la metodologia usata nel paper (VAR, causalità di
Granger parametrica e non parametrica, modello di volatilità GJR-BEKK con
test di Wald sullo spillover) tramite un simulatore che genera dati con una
struttura di causalità nota a priori, per verificare se le procedure di
stima/test del paper la recuperano correttamente.

## Informazioni progetto

| Campo           | Valore                     |
| --------------- | -------------------------- |
| Corso           | `Internet of Things - IoT` |
| Studente        | `Luca Camiolo`             |
| Matricola       | `1000084790`               |
| Anno Accademico | `2025/2026`                |

## Struttura della repository

```
.
├── relazione.pdf <- relazione del progetto
├── presentazione.pptx <- powerpoint
└── simulatore/
    ├── README.md              <- documentazione tecnica del simulatore
    ├── dgp.py                 <- data-generating process (VAR(1) + GJR-BEKK(1,1))
    ├── granger_tests.py       <- causalità di Granger, parametrica e non parametrica
    ├── bekk_model.py          <- stima MLE del GJR-BEKK(1,1) + test di Wald
    ├── run_experiment.py      <- pipeline sperimentale end-to-end
    ├── requirements.txt
    └── example_report.md      <- output di esempio già generato
```

## Come riprodurre i risultati

```bash
cd simulatore
pip install -r requirements.txt

# report completo su dati simulati (~8 secondi)
python run_experiment.py --n-obs 1200 --seed 42 --out example_report.md

# verifica di potenza/dimensione dei test su più ripetizioni
python run_experiment.py --monte-carlo 30 --n-obs 800 --out monte_carlo.md
```

Un output di esempio, già generato, è in `simulatore/example_report.md`.

## Limiti metodologici dichiarati

- Il test di causalità non parametrico usa un bootstrap stazionario per il
  p-value al posto della formula di varianza asintotica di Diks e Panchenko
  (2006), più complessa da riprodurre fedelmente.
- La stima del modello GJR-BEKK (15 parametri) è identificata solo a meno del
  segno simultaneo di riga/colonna delle matrici dei coefficienti; si vincola
  il segno degli elementi diagonali per rimuovere l'ambiguità.
- Con campioni di dimensione modesta (poche centinaia/migliaia di
  osservazioni) la potenza dei test, in particolare quello sullo spillover
  asimmetrico, può essere limitata — verificabile con la modalità Monte Carlo.

Discussione completa di metodologia, risultati e limiti nella relazione.

## Riferimenti

Okorie, D.I., Gnatchiglo, J.M., Wesseh, P.K. (2024). Electricity and
cryptocurrency mining: An empirical contribution. _Heliyon_, 10, e33483.
