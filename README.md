# Federated Learning Simulation
**Problem**

Tradycyjne uczenie maszynowe wymaga zebrania wszystkich danych w jednym miejscu. W praktyce często jest to niemożliwe (np. dane medyczne, bankowe), a dodatkowo różni klienci mogą mieć niezbalansowane zbiory danych, co obniża skuteczność modelu.
Potrzebne są metody, które pozwalają trenować wspólne modele bez udostępniania surowych danych. Poniższy kod został przygotowany głównie do zbadania problemu występowania niezbalansowanych danych, które tendencję do faworyzowania klasy dominującej, co prowadzi do sytuacji, w której model dobrze przewiduje klasę dominującą, ale ma trudności z prawidłową klasyfikacją rzadziej występujących klas.
W konsekwencji, model staje się mniej skuteczny w realnych zastosowaniach, gdzie istotna jest dokładność dla wszystkich klas.

**Rozwiązanie**

Poniższy kod został użyty do przeprowadzenia eksperymetów badających możliwości wykorzystania uczenia federacyjnego do radzenia sobie z problemem niezbalansowanych
danych oraz analizy wpływ wybranych, metod agregacji na poprawę jakości modeli federacyjnych w sytuacjach, gdy dane są niezbalansowane. Efektywność podejść została oceniona na podstawie następujących miar: dokładność (accuracy), swoistość (specificity) oraz czułość (sensitivity). 
Analiza tych miar pozwoliła ocenić jakość modeli predykcyjnych, zarówno pod kątem ich ogólnej skuteczności, jak i zdolności do prawidłowego rozpoznawania mniej licznych klas.

**Kroki implemenacji**

* Stworzyłam środowisko do eksperymentów z federated learning przy użyciu frameworka Flower i PyTorch.

* Dane klientów symulowałam przy użyciu oddzielnych plików CSV (client1.csv, client2_unbalanced_7_3.csv, client2_unbalanced_8_2.csv, client3.csv).

* Dane pochodziły ze zbiorów dotyczących raka piersi (breast cancer dataset).

* Zaimplementowałam klasyczną sieć feed-forward neural network (3 warstwy) dla binarnej klasyfikacji.

* Zaimplementowałam dwa podejścia:

  * Tradycyjne uczenie scentralizowane (baseline w pliku no_federated_learning.py).

  * Federated Learning (FL) z agregacją parametrów między klientami (federated_learning.py).

* Dodałam różne strategie agregacji, m.in.:
  * weighted average (FedAvg),
  * median aggregation,
  * weighted median aggregation.

* W trakcie treningu monitorowałam metryki: loss, accuracy, sensitivity (czułość), specificity (specyficzność).

* Wyniki wizualizowałam na wykresach (zmiana metryk w kolejnych rundach).


🧰 Stack technologiczny

* Python, PyTorch, Flower (Federated Learning Framework)

* NumPy, Pandas, scikit-learn

* Matplotlib (wizualizacja metryk)
