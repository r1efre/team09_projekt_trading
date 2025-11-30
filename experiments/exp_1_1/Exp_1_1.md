# Team 09 - Projekt Trading

### Problem Definiton:

**Target**

Ziel dieses Projekts ist die Vorhersage der Trendrichtung des Bitcoin-Preises in der nächsten Stunde.
Für die Modellierung werden stündliche Bitcoin-Daten im Zeitraum 01.01.2021 bis 01.11.2025 als Trainings- und Validierungsgrundlage verwendet.
Auf Basis dieser historischen Stundenwerte soll das Modell lernen, für jeden Zeitpunkt vorherzusagen, ob der Bitcoinpreis in der darauffolgenden Stunde steigt, fällt oder innerhalb eines definierten Schwellenwerts neutral bleibt.
Die Trendrichtung wird dabei anhand des prozentualen Preisreturns zwischen dem aktuellen und dem nachfolgenden Schlusskurs berechnet.
Bewegungen innerhalb einer kleinen Toleranzzone werden als neutral klassifiziert, um Marktrauschen zu reduzieren und stabile Labels zu erzeugen.

**Input features**

Das Modell verarbeitet pro Stunde eine Reihe Features, die Preisstruktur, Trend, Momentum und Cross-Asset-Information abbilden:
- Preis- und Volumenmerkmale: open, high, low, close, volume, VWAP (volumengewichteter Durchschnittspreis)
- Momentum-Merkmale: return_1h, return_6h, return_24h --> prozentuale Preisveränderung über 1, 6 Stunden bzw. 24 Stunden
- Trendindikatoren: EMA_6 und EMA_24 --> normalisierte exponentielle gleitende Durchschnitte
- RSI (Trendstärke-Indikator aus Preisänderungen)
- ATR (Maß für die Volatilität des Marktes)
- Cross-Asset-Features anhand von Ethereum: ETH_Close, ETH_return_1h, ETH_return_6h, ETH_return_24h, ETH/BTC Ratio (relative Stärke zwischen ETH und BTC)

### Procedure Overview:

- Datensammlung: Erhebung von stündlichen Bitcoin- und Ethereum-Daten im Zeitraum 01.01.2021 bis 01.11.2025
- Feature Engineering: Berechnung aller oben beschriebenen preis-, volumen-, trend- und momentumbezogenen Merkmale sowie Cross-Asset-Features.
- Labelgenerierung: Berechnung des 1-Stunden-Returns und Klassifikation der Trendrichtung in Up / Down / Neutral anhand einer Toleranzschwelle.
- Modelltraining: Training eines LSTM-Netzwerks, das aus 48-stündigen Sequenzen die Trendklasse der nächsten Stunde vorhersagt.

Zielsetzung: 
Die Analyse soll Muster in Preis-, Volumen- und Cross-Asset-Beziehungen identifizieren, die kurzfristige Trendbewegungen im Bitcoin-Markt vorhersagen.
Durch die Modellierung der Trendrichtung (Up/Down/Neutral) auf Basis historischer Daten sollen Entscheidungsgrundlagen geschaffen werden, um günstige Kauf- und Verkaufszeitpunkte zu erkennen und so potenziell profitablere Handelsentscheidungen zu ermöglichen.

In einem hypothetischen Trading-Kontext könnten diese Signale wie 
folgt interpretiert werden:
- Up → Kaufsignal
- Down → Verkaufssignal  
- Neutral → Keine Aktion (Position halten)


## Step 1 - Data Acquisition

Es werden Rohmarktdaten für Bitcoin und Ethereum extrahiert. Da Kryptowährungen rund um die Uhr gehandelt werden, ist keine Filterung nach Öffnungs- oder Schließzeiten erforderlich.

**Script**

[scripts/01_data_acquisition/01_data_acquisition.py](scripts/01_data_acquisition/01_data_acquisition.py)

Lädt stündliche, Kursbalken (OHLCV) für Bitcoin und Ethereum im Zeitraum vom 01.01.2024 bis zum 01.11.2025 und speichert die Daten als Parquet-Dateien.
Im nächsten Schritt werden die Bitcoin-Daten mit dem Ethereum-Schlusskurs über den gemeinsamen Zeitstempel gemergt, sodass der ETH-Close-Wert als zusätzliches Feature zur Verfügung steht.

Bar data example:

![Datenübersicht](images/DatenUebersicht.png) 

---

## Step 2 - Data Understanding

Analyse historischer BTC-Preisdaten, um die Struktur, Eigenschaften und das Verhalten der Zeitreihe zu verstehen.
Ziel ist es, wichtige statistische Abhängigkeiten, sich wiederholende Muster und Besonderheiten zu identifizieren, die die Auswahl der Merkmale und die Architektur des Modells beeinflussen.


Bei der Datenprüfung konnten keine fehlenden Werte (NA/NaN) festgestellt werden.

**Data Columns**

| Column              | Description                                                                            |
|---------------------|----------------------------------------------------------------------------------------|
| timestamp           | Der genaue Zeitpunkt (UTC), zu dem die stündliche Kerze geschlossen wurde - Zeitindex. |
| open (float)        | Der BTC-Preis zu Beginn des Stundenintervalls.                                         |
| high (float)        | Der höchste BTC-Preis, der während dieser Stunde erreicht wurde.                       |
| low (float)         | Der niedrigste BTC-Preis, der während dieser Stunde erreicht wurde.                    |
| close (float)       | Der BTC-Preis am Ende des Stundenintervalls (Zielvariable).                            |
| volume (float)      | Das gehandelte BTC-Volumen während der Stunde - Marktaktivität.                        |
| trade_count (float) | Anzahl der ausgeführten Trades innerhalb einer Stunde.                                 |
| vwap (float)        | Volumengewichteter Durchschnittspreis für die Stunde.                                  |
| eth_close (float)   | Der stündliche Schlusskurs von Ethereum (ETH).                                         |


**Descritive Statistics**

| Statistic    | open          | high          | low           | close          | volume       | trade_count   | vwap           | eth_close    |
|--------------|---------------|---------------|---------------|----------------|--------------|---------------|----------------|--------------|
| count        | 42339.000000  | 42339.000000  | 42339.000000  | 42339.000000   | 42339.000000 | 42339.000000  | 42339.000000   | 42339.000000 |
| mean         | 53085.164609  | 53295.084529  | 52861.723574  | 53084.377853   | 78.447927    | 1775.976806   | 52159.513397   | 2513.903310  |
| std          | 29053.359436  | 29126.455657  | 28982.046140  | 29052.349537   | 180.463969   | 3234.410173   | 29774.942871   | 920.348506   |
| min          | 15627.650000  | 15750.440000  | 8200.000000   | 15631.840000   | 0.000000     | 0.000000      | 0.000000       | 722.000000   |
| 25%          | 29218.204031  | 29298.214064  | 29133.980952  | 29216.630861   | 0.015927     | 5.000000      | 28437.060330   | 1771.182339  |
| 50% (median) | 44716.370000  | 44988.310000  | 44444.480000  | 44728.140000   | 7.310082     | 530.000000    | 43825.937372   | 2405.479000  |
| 75%          | 66949.332500  | 67166.909250  | 66729.690465  | 66939.260000   | 72.199770    | 2395.500000   | 66653.792593   | 3231.658500  |
| max          | 126093.044500 | 126262.032000 | 125280.720000 | 126117.150000  | 5213.685947  | 110487.000000 | 125566.851844  | 4933.850000  |

Die Werte zeigen eine große Bandbreite sowohl bei den BTC-Preisschwankungen als auch bei den Volumen- und Handelsaktivitätsindikatoren.

- Preise (OHLC): der durchschnittliche BTC-Preis liegt bei etwa 53.000 USD, jedoch mit einer sehr hohen Standardabweichung von etwa 29.000 USD, was auf starke langfristige Marktbewegungen hindeutet. Der Tiefstwert von 8.200 US-Dollar und der Höchstwert von über 126.000 US-Dollar zeigen eine enorme Volatilität im betrachteten Zeitraum.
- Volumen und Trade Count: das Handelsvolumen weist eine stark asymmetrische Verteilung auf: der Median liegt bei nur 7,31, während der Höchstwert bei über 5200 liegt. Auch die Anzahl der Transaktionen variiert stark, und zwar von 0 bis zu über 110.000 pro Stunde. Dies deutet auf große Unterschiede in der Marktaktivität zwischen ruhigen und hochvolatilen Phasen hin.
- VWAP: der durchschnittliche VWAP liegt bei 52.159 USd und schwankt ähnlich wie die anderen Preisvariablen, was eine stabile Preisstruktur bestätigt.
- ETH Close: der durchschnittliche Schlusskurs von ETH liegt bei etwa 2514 US-Dollar, ebenfalls mit einer hohen Streuung. Dies bestätigt, dass auch der ETH-Markt während des Analysezeitraums starken Schwankungen unterworfen war.

Insgesamt zeigen die Werte eine hohe Volatilität, große Unterschiede in der Marktaktivität und starke Preisschwankungen. Diese Merkmale sind für die nachfolgende Modellierung wichtig und bestätigen die Zweckmäßigkeit der Wahl des LSTM-Modells.

**Script**

[scripts/02_data_understanding/plotter.py](scripts/02_data_understanding/plotter.py)

**Plots**

*1) Entwicklung des BTC-Schlusskurses über den gesamten Zeitraum 2024–2025 vs 2021-2025*

Dieses Diagramm zeigt die langfristige Entwicklung des Schlusskurses von BTC über den gesamten Beobachtungszeitraum. Es ermöglicht die Darstellung der Gesamtstruktur einer Zeitreihe.

![Schlusskurse 2024-2025](images/02_btc_close_2024-2025.png) 

Ursprünglich haben wir einen Chart mit den Schlusskursen von BTC nur für den Zeitraum 2024–2025 erstellt.
In diesem Zeitraum war jedoch fast ausschließlich ein anhaltender Aufwärtstrend ohne nennenswerte Korrekturen zu beobachten.
Damit das Modell nicht nur auf Wachstumsphasen des Marktes trainiert werden konnte, haben wir den Zeitrahmen auf 2021–2025 erweitert.

![Schlusskurse 2021-2025](images/02_btc_close_2021-2025.png) 

Dieser Bereich umfasst starke Preisrückgänge, beispielsweise den starken Einbruch, der im Dezember 2021 einsetzte, wodurch das Modell verschiedene Marktbedingungen berücksichtigen kann.

*2) Durchschnittlicher BTC-OHLC-Verlauf und Handelsvolumen pro Stunde (auf Basis des letzten Jahres)*

Wir untersuchen den Durchschnittswert von BTC-OHLC für das letzte Jahr in stündlichen Intervallen. Außerdem wird das durchschnittliche Handelsvolumen pro Stunde angezeigt.
Das Diagramm hat zwei Y-Achsen:
- linke Achse: Preis (Eröffnung, Maximum, Minimum, Schlusskurs)
- rechte Achse: durchschnittliches Handelsvolumen Die X-Achse zeigt die Zeit an, wobei jeder Punkt den Durchschnittswert für diese Stunde während des gesamten Jahres darstellt.


![OHLC-Verlauf](images/02_btc_ohlcv_mean_intraday.png)

Es ist ersichtlich, dass sich die Eröffnungs- und Schlusskurse relativ stabil entwickeln, ohne große Schwankungen zwischen den Stunden. Die Höchst- und Tiefstkurse weisen jedoch zu bestimmten Tageszeiten erhebliche Schwankungen auf. Das Handelsvolumen korreliert teilweise mit diesen Schwankungen, steigt jedoch nicht immer synchron an.
Eine große Bandbreite von Höchst- und Tiefstwerten bedeutet eine erhöhte Volatilität, die oft ein Indikator für eine Trendwende oder stärkere Bewegungen ist.


*3) Durchschnittlicher stündlicher High–Low-Range von BTC (auf Basis des letzten Jahres)*

Dieser Graph zeigt den durchschnittlichen Unterschied zwischen High und Low („Range“) pro Stunde. Jede Markierung steht für eine Stunde des Tages und zeigt, wie volatil der BTC-Preis im Durchschnitt ist.

![High-Low-Range](images/02_btc_high-low_range.png)

Es gibt eine deutlich erkennbare Volatilitätsphase zwischen 13:00 und 17:00 Uhr. Der Bereich nimmt dort stark zu und erreicht sein Tageshoch, bevor er nach 18:00 Uhr wieder abnimmt. Dies spiegelt erneut die hohe Volatilität des Marktes wider.
In Stunden mit einem hohen durchschnittlichen Bereich ist die Wahrscheinlichkeit, dass der Schlusskurs deutlich fällt oder steigt, höher,  d. h. es gibt ein stärkeres Signal für die Klassifizierung „aufwärts/abwärts”.


*4) Veränderung der BTC/ETH-Close-Preise über den letzten Tag*

Das Diagramm zeigt den durchschnittlichen stündlichen Schlusskurs (Close) für BTC und ETH in den letzten 30 Tagen.
Das Diagramm hat zwei Y-Achsen:
- links: BTC Close
- rechts: ETH Close Die Punkte geben den Durchschnittswert pro Stunde an.

![BTC/ETH-Close-Preise](images/02_btc_eth_close_together.png)

BTC und ETH weisen sehr ähnliche Kurven auf. Beide Kritowährungen steigen und fallen oft synchron.
Die starke Parallelität deutet auf eine hohe Korrelation zwischen BTC und ETH hin. 
Dies deutet darauf hin, dass die ETH-Daten als zusätzliche Funktionen tatsächlich nützlich sind, um das LSTM-Modell zu verbessern.

*5) Korrelation zwischen BTC und ETH (letzte 30 Tage)*

Dieses Diagramm ist ein Streudiagramm, in dem jeder Punkt ein Wertepaar aus BTC_close und ETH_close zur gleichen Uhrzeit darstellt.
Wir haben dieses Diagramm erstellt, um endgültig zu bestätigen, dass ETH ein nützliches zusätzliches Merkmal für das Modell ist.

![Korrelation zwischen BTC und ETH](images/02_btc_eth_correlation.png)

Auf dem Diagramm ist deutlich zu erkennen, dass die Datenpunkte entlang einer aufsteigenden Linie konzentriert sind.
Dies deutet auf eine sehr starke positive lineare Abhängigkeit zwischen BTC und ETH hin.
Der berechnete Pearson-Korrelationskoeffizient r = 0,96 bestätigt diese Abhängigkeit zusätzlich.

---

## Step 3 - Data Preparation (Pre-Split)

**Script**

- [scripts/03_pre_split_prep/features.py](scripts/03_pre_split_prep/features.py)
- [scripts/03_pre_split_prep/targets.py](scripts/03_pre_split_prep/targets.py)
- [scripts/03_pre_split_prep/main.py](scripts/03_pre_split_prep/main.py)
- [scripts/03_pre_split_prep/plot_features.py](scripts/03_pre_split_prep/plot_features.py)

### Features Berechnung

Folgende Features werden in dem Skript features.py berechnet. 

**Data Columns**

| Column                   | Description                                                                                                                                    |
|--------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|
| btc_return (1h, 6h, 24h) | Prozentualer Preisänderung des Close Preises in den letzten 6h, 24h oder der letzten Stunde.                                                   |
| eth_return (1h, 6h, 24h) | Prozentualer Preisänderung des Close Preises von Ethereum in den letzten 6h, 24h oder der letzten Stunde.                                      |
| eth_btc_ratio            | Relative Stärke von Ethereum Close Preis gegenüber Bitcoin Close Preis                                                                         |
| ema_6 und ema_24         | Exponentiell gewichteter gleitender Durchschnitt, wobei neuere Close Preise von Bitcoin mehr Gewicht bekommen                                  |
| rsi                      | Momentum-Indikator, berechnet über die letzten 24 Stunden -> zeigt on Markt überkauft oder überverkauft ist                                    |
| atr_24                   | Volatilität -> Misst, wie stark sich der Bitcoin-Preis über die letzten 24 Stunden durchschnittlich bewegt hat, relativ zum aktuellen Preis.   |
                                                           

### Target Berechnung

Das Projekt hat das Ziel, die Trendrichtung des Bitcoin-Schlusskurses für die jeweils nächste Stunde vorherzusagen. Die Trendrichtung wird dabei in drei Klassen eingeteilt:

- UP: 1
- Neutral: 0
- DOWN: -1

Der Trendwert in einer Zeile zum Zeitpunkt t beschreibt die Kursbewegung von t bis t+1. Die Klassifizierung erfolgt auf Grundlage des stündlichen Returns btc_return_1h, der in der Zeile t+1 steht.

- Ist btc_return_1h(t+1) > 0, wird die Klasse UP vergeben.
- Ist btc_return_1h(t+1) < 0, wird die Klasse DOWN vergeben.
- Liegt der Wert innerhalb einer definierten Toleranzzone, wird die Klasse NEUTRAL vergeben.

![Trend Berechnung](images/trendBerechnung.png)

Die Toleranzzone ist dabei nicht statisch, sondern wird dynamisch an die aktuelle Marktvolatilität angepasst.
Dazu wird die typische Tagesvolatilität anhand des ATR(24h)-Indikators geschätzt.

Ein Trend wird als neutral klassifiziert, wenn:
∣btc_return_1h(t+1)∣ < 0.25×ATR24​(t)

Das bedeutet: Bewegungen, die weniger als 25 % der typischen täglichen Volatilität ausmachen, gelten als marktübliches Rauschen und werden nicht als richtungsstarker Trend gewertet.

Nach der Features und Target Generierung werden NaN Werte entfernt, die durch das Feature Engineering entstanden sind.

### Deskriptive Statistiken

![Deskriptive Statistiken](images/03_Deskriptive_Statistiken.png)

- BTC- und ETH-Returns haben kleine Mittelwerte (≈ 0) 
- Die Standardabweichung steigt mit längeren Zeitfenstern (1h < 6h < 24h) → je größer das Zeitintervall, desto größer die erwartete Schwankung.
- ETH ist volatiler als BTC → die Standardabweichungen von ETH-Returns sind in jedem Zeitfenster höher als bei BTC.
- Die Return-Verteilungen zeigen deutliche Ausreißer (z. B. bei 24-h-Returns: max ≈ +0.23, min ≈ −0.20), was auf starke Marktbewegungen hinweist
- Der RSI liegt im Durchschnitt bei ~50, was bedeutet, dass der Markt über den Gesamtzeitraum weder signifikant überkauft (>70) noch überverkauft (<30) war
- Indikatoren wie EMA (6h/24h) haben hohe Werte, da sie direkt auf dem BTC-Preis basieren, was ihre Skalierung von Returns klar unterscheidet

### Visualisierung 

**Plots**

*1) Trendverteilung des Bitcoin-Marktes im Zeitverlauf (2-Monats-Intervalle)*

![Trend Verteilung](images/03_Trend_Verteilung_pro_2Monate.png)

- Jede Säule zeigt, wie viele Stunden innerhalb eines 2-Monats-Fensters als UP, DOWN oder NEUTRAL klassifiziert wurden.
- Die Verteilung bleibt über die Jahre stabil
- keine strake Dominanz einer bestimmten Trendklasse
- Beim Splitten der Daten sollten alle Trendklassen sowohl im Trainings- als auch im Validierungsdatensatz ausreichend vertreten sein

*2) Entwicklung der Bitcoin-Volatilität (ATR-24h) von 2021 bis 2025)*

![Bitcoin-Volatilität](images/03_Zeitreihenplot_atr24.png)

- Zeigt die tägliche prozentuale Volatilität von Bitcoin gemessen über ein 24-Stunden-Fenster
- Hohe Ausschläge markieren Phasen starker Marktbewegungen, während ruhige Perioden niedrige ATR-Werte zeigen.
- Die Grafik verdeutlicht, dass die Volatilität langfristig abnimmt, mit vereinzelten starken Peaks

*3) Zusammenhang zwischen BTC- und ETH-Returns*

![BTC ETH Return Korrelation](images/03_return_correlation.png)

- Zeigt die lineare Beziehung zwischen stündlichen BTC- und ETH-Returns.
- Die deutliche Aufwärtswolke und der Pearson-Korrelationswert (r = 0.83) deuten auf eine sehr starke positive Korrelation hin.
- Daraus lässt sich schließen: Steigt BTC, steigt ETH typischerweise ebenfalls.

*4) Rolling Lag-Korrelation zwischen ETH und BTC Returns (zeitversetzt um 1 Stunde)*

![BTC ETH Return Korrelation zeitversetzt](images/03_eth_btc_corr_zeitversetzt.png)

- Zeigt, ob vergangene ETH-Returns (lagged um 1h) Bewegungen im BTC-Return vorhersagen können.
- Die Korrelation bleibt über alle Jahre sehr nahe bei 0 – kaum systematischer Zusammenhang erkennbar.
- Positive oder negative Ausreißer treten nur kurzfristig auf und wirken zufällig, nicht strukturell.
- ETH Return nicht hilfreich bei der Vorhersage der zuküngtigen BTC Entwicklung

*5) Rolling Lag-Korrelation zwischen BTC und BTC Returns (zeitversetzt um 1 Stunde)*

![BTC BTC Return Korrelation zeitversetzt](images/03_btc_btc_corr_zeitversetzt.png)

- Zeigt, wie stark der BTC-Return der letzten Stunde mit dem BTC-Return der aktuellen Stunde zusammenhängt.
- Die Korrelation schwankt um 0, was zeigt, dass Vergangenheits-BTC kaum Einfluss auf die nächste Stunde hat.
- Es gibt keine stabilen positiven oder negativen Muster, daher liefern lagged BTC-Returns keine sinnvolle Vorhersagekraft.
- Die gleiche Analyse wurde durchgeführt, wo BTC-Return um 12 und um 24 Stunden zurückversetzt wurde. Auch bei diesen Analysen ließen sich keine Zusammenhänge feststellen.

*6) EMA-Differenz (6h–24h) und BTC-Return (1h) im Vergleich über die letzten 12 Monate*

![EMA-Differenz und BTC-Return](images/03_ema_dif.png)

- EMA-Differenz als Trendindikator: Positive Werte zeigen kurzfristige Aufwärtsdynamik, negative Werte eine kurzfristige Trendabschwächung.
- BTC-Returns folgen ähnlichen Mustern: Geglättete Returns bewegen sich häufig in dieselbe Richtung wie die EMA-Differenz
- Zusammenhang zwischen EMA-Differenz und BTC-Return erkennbar

---

## Step 3 - Split Data

**Script**

[scripts/04_split_data/split.py](scripts/04_split_data/split.py)


Die Daten werden aufgeteilt in:
- Trainingsdaten (ca. 70% der Daten) - 2021-01-02 bis 2024-05-19
- Validierungsdaten (ca. 20% der Daten) - 2024-05-20 bis 2025-05-07
- Testdaten (ca. 10% der Daten) - 2025-05-08 bis 2025-10-31



