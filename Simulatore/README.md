# Simulatore VAR / Granger causality / GJR-BEKK

Simulatore del modello econometrico usato in:

> Okorie, D.I., Gnatchiglo, J.M., Wesseh, P.K. (2024). *Electricity and
> cryptocurrency mining: An empirical contribution.* Heliyon 10, e33483.

L'articolo studia la relazione tra il mercato dell'elettricità e i mercati
delle criptovalute (Bitcoin, Ethereum) usando tre strumenti: un modello VAR(p)
per la media condizionata, test di causalità di Granger parametrici e non
parametrici, e un modello di volatilità BEKK asimmetrico (GJR-BEKK) con test
di Wald per lo spillover di volatilità.

## 1. Dove si trova la parte sperimentale/quantitativa nell'articolo

Sezione 4, "Results and discussion", con le Tabelle 1-6:

| Tabella | Contenuto | Modulo che la riproduce |
|---|---|---|
| Table 1 | Statistiche descrittive, Jarque-Bera, Ljung-Box su rendimenti/hash rate | `run_experiment.py::descriptive_table` |
| Table 2 | Test ADF di stazionarietà su prezzi, hash rate, indici elettricità | `run_experiment.py::stationarity_table` |
| Table 3 | Causalità di Granger (parametrica e non parametrica) tra prezzo crypto e hash rate | `granger_tests.py` |
| Table 4 | Causalità di Granger tra rendimento crypto e domanda elettrica, per quantile di costo elettrico | `granger_tests.py` |
| Table 5 | Come Table 4 ma su indice elettrico globale | `granger_tests.py` |
| Table 6 | Test di Wald sullo spillover di volatilità dal modello BEKK-GJR | `bekk_model.py` |

Le equazioni implementate sono quelle del paper (Sezione 3.2):
- equazioni (1)-(2): equazione media VAR(p), `phi_i(L,p_i) A_it = b_i0 + b_i1*t + a_it`
- equazioni (3)-(5): covarianza condizionata BEKK-GJR, `H_t = C'C + X'a_{t-1}a_{t-1}'X + Y'H_{t-1}Y + Z'eta_{t-1}eta_{t-1}'Z`
- ipotesi H01-H04 (Sezione 3.2): causalità di Granger parametrica e non parametrica
- ipotesi sui test di Wald (Sezione 3.2 / Table 6): `x21=y21=0`, `x12=y12=0`, `z12=z21=0`

## 2. Perché un "simulatore" e non i dati originali del paper

I dati originali (prezzi da Coin Market Cap, hash rate da Quandl/Etherscan,
domanda elettrica dalla World Bank WDI, 2013-2019) non sono allegati
all'articolo e non sono scaricabili da questo ambiente. Il software quindi:

1. **Simula** dati coerenti con il *data generating process* del paper
   (`dgp.py`): un VAR(1) per la media più un GJR-BEKK(1,1) per la volatilità,
   con parametri di causalità/spillover impostabili a piacere e **noti a
   priori**. Questo permette di verificare se le procedure di stima/test del
   paper (`granger_tests.py`, `bekk_model.py`) recuperano correttamente la
   struttura vera - il modo standard di validare una procedura econometrica
   prima di usarla su dati reali e rumorosi.
2. **Accetta anche dati reali**, tramite `--csv`, per chi vuole applicare la
   stessa pipeline a una propria serie storica di rendimenti crypto e di un
   proxy della domanda elettrica.

## 3. Struttura del progetto

```
dgp.py             Data-generating process: VAR(1) + GJR-BEKK(1,1) (eq. 1-5)
granger_tests.py    Causalità di Granger parametrica (VAR/F-test) e non
                     parametrica (statistica in stile Diks-Panchenko 2006)
bekk_model.py        Stima MLE del GJR-BEKK(1,1) (JIT via numba) + test di Wald
run_experiment.py    Pipeline sperimentale end-to-end (Tabelle 1,2,3/5,6) +
                     modalità Monte Carlo per stimare potenza/dimensione dei test
requirements.txt
```

## 4. Uso

```bash
pip install -r requirements.txt

# Report completo su dati simulati (parametri di default in dgp.py)
python run_experiment.py

# Cambiare dimensione campionaria e seed
python run_experiment.py --n-obs 1500 --seed 7 --out results.md

# Applicare la pipeline a dati propri (CSV a 2 colonne)
python run_experiment.py --csv miei_dati.csv --col-crypto btc_ret --col-elec elec_growth

# Studio Monte Carlo: potenza/dimensione empirica dei test su N ripetizioni
python run_experiment.py --monte-carlo 30 --n-obs 800 --out monte_carlo.md
```

Ogni esecuzione produce un report Markdown (`results.md` di default) con le
tabelle descritte sopra, stampato anche a schermo.

## 5. Note metodologiche e limiti (onestà scientifica)

- **Test non parametrico**: il paper usa la statistica di Diks & Panchenko
  (2006), la cui varianza asintotica in forma chiusa richiede stime di
  integrali a 6 dimensioni non banali da riprodurre esattamente. Qui si
  calcola la stessa statistica basata su rapporti di densità locali (kernel
  indicatore, bandwidth e dimensione di embedding di default = quelle
  riportate nelle note delle tabelle del paper: 0.5 e 2), ma il p-value è
  ottenuto con un **bootstrap stazionario** (block-shuffling della serie
  "causa" per distruggere la dipendenza incrociata mantenendo
  l'autocorrelazione propria di ciascuna serie) invece della formula
  asintotica. È una sostituzione standard e trasparente, esatta a meno
  dell'errore Monte Carlo nel numero di repliche bootstrap. Essendo un test
  O(n^2), va applicato a un sotto-campione (parametro
  `--nonparam-subsample`, default 250 osservazioni).
- **Stima del BEKK-GJR**: il modello ha 15 parametri liberi ed è
  identificato solo a meno del segno simultaneo di riga/colonna di C, X, Y, Z
  (la verosimiglianza dipende solo dalle forme quadratiche). Per rimuovere
  questa ambiguità si vincola il segno degli elementi diagonali
  ("effetto proprio") a essere non negativo; i termini di spillover
  fuori diagonale restano liberi nel segno. La stima usa un multi-start (4
  punti di partenza) con verosimiglianza compilata JIT (`numba`) per
  restare veloce nonostante il multi-start. Anche così, la superficie di
  verosimiglianza resta relativamente piatta vicino all'ottimo: l'ottimizzatore
  può segnalare `converged: False` pur avendo la log-verosimiglianza
  stabilizzata fra i vari punti di partenza - un comportamento tipico per
  questa classe di modelli e non necessariamente un errore. Gli errori
  standard vengono dall'Hessiana numerica della log-verosimiglianza negativa.
- **Potenza dei test**: la modalità `--monte-carlo` mostra che, con
  campioni di dimensione realistica (poche centinaia/migliaia di
  osservazioni), il test di Wald sullo spillover asimmetrico (Z) e talvolta
  anche quello sullo spillover ARCH/GARCH (X, Y) hanno potenza limitata:
  è lo stesso tipo di difficoltà pratica nella stima dei modelli BEKK
  multivariati discussa in letteratura, e uno dei motivi per cui il paper
  usa quasi 7 anni di dati giornalieri (oltre 2000 osservazioni) invece di
  poche centinaia.
- **numba è opzionale**: se non installato, `bekk_model.py` ricade
  automaticamente su una funzione Python pura (più lenta di circa 2000x);
  il fit resta corretto ma diventa impraticabile per campioni superiori a
  poche centinaia di osservazioni.
