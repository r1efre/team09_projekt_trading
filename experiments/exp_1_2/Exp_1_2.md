# Team 09 - Projekt Trading

## 📚 Table of Contents

- [Ergebnisse erstes Experiment](#ergebnisse-erstes-experiment)
- [Problem Definiton](#problem-definiton)
- [Step 1 - Data Acquisition](#step-1---data-acquisition)
- [Step 2 - Data Understanding](#step-2---data-Understanding)
- [Step 3 - Data Preparation (Pre-Split)](#step-3---data-preparation-pre-split)
- [Step 4 - Split Data](#step-4---split-data)
- [Step 5 - Post-Split Preparation](#step-5---post-split-preparation)
- [Step 6 - Feature Selection](#step-6---feature-selection)
- [Step 7 - Model Training](#step-7---model-training)
- [Step 8 - Model Testing](#step-8---model-testing)
- [Step 9 - Model Deployment](#step-9---deployment)
- [Fazit und Next Steps](#fazit-und-next-steps)

---

### Ergebnisse erstes Experiment

--- 

### Problem Definiton:

**Target**

In diesem zweiten Experiment wird die Richtung der kurzfristigen Bitcoin-Preisentwicklung vorhergesagt. Ziel ist es, zu bestimmen, ob der Bitcoin-Preis innerhalb der nächsten 60 Minuten steigt oder fällt.
Im Gegensatz zum ersten Experiment, das auf stündlichen Daten basierte, werden in diesem Experiment minütliche Preisdaten im Zeitraum vom 01.01.2024 bis zum 30.11.2025 als Trainings- und Validierungsdaten verwendet. Durch die höhere zeitliche Auflösung soll eine feinere Erfassung kurzfristiger Marktbewegungen ermöglicht werden.
Während im ersten Experiment drei Klassen (steigend, fallend, neutral) betrachtet wurden, wird in diesem Experiment bewusst auf die Vorhersage eines neutralen Trends verzichtet. Stattdessen wird das Problem als binäre Klassifikationsaufgabe formuliert, um die Modellkomplexität zu reduzieren und die Trennschärfe zwischen steigenden und fallenden Marktbewegungen zu erhöhen.
Die Trendrichtung wird anhand des prozentualen Preisreturns zwischen dem aktuellen Zeitpunkt t und dem Schlusskurs nach 60 Minuten (t + 60) bestimmt. Ein positiver Return wird als steigender Trend, ein negativer Return als fallender Trend interpretiert.

Dieses Experiment basiert auf:
[Erstes Experiment](../exp_1_1/Exp_1_1.md)

**Input features**

Das Modell verarbeitet minütlich aufgelöste Zeitreihendaten, die verschiedene Aspekte der kurzfristigen Marktdynamik abbilden. Die verwendeten Features lassen sich in Preis- und Volumenmerkmale, Momentum- und Trendindikatoren sowie Cross-Asset-Informationen unterteilen.
Da nun minütliche und nicht stündliche Daten verwendet werden, werden die Zeithorizonte über denen die Features berechnet werden angepasst.

- Preis- und Volumenmerkmale: open, high, low, close und volume jeweils auf Minutenbasis.
- Momentum-Merkmale: return_5min, return_30min, return_60min --> prozentuale Preisveränderung über 5, 30 bzw. 60 Minuten
- Trendindikatoren: EMA_15 und EMA_60 --> normalisierte exponentielle gleitende Durchschnitte
- RSI (Trendstärke-Indikator aus Preisänderungen) --> berechnet über die letzten 14 Perioden
- ATR (Maß für die Volatilität des Marktes) --> berechent über die letzten 14 Perioden
- Cross-Asset-Features anhand von Ethereum: ETH_Close, ETH_return_5min, ETH_return_30min, ETH_return_60min, ETH/BTC Ratio (relative Stärke zwischen ETH und BTC)