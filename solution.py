import sys
from math import log

args = sys.argv

def CitajIzDatoteke(path):
    with open(path, 'r', encoding="utf-8") as file:
        return file.readlines()

class ID3:
    def __init__(self, stablo, skup_x, skup_y, max_depth=None, prefix='', level=1):
        self.stablo = stablo
        self.skup_x = skup_x
        self.skup_y = skup_y
        self.max_depth = max_depth
        self.prefix = prefix
        self.level = level
        self.tree = {}

    def fit(self):
      if self.max_depth == 0:
        leaf_label = max(sorted(self.skup_y), key=self.skup_y.count)
        print(f"{self.prefix} {leaf_label}")
        self.tree = leaf_label
      else:
        self.tree = self.fit2(self.stablo, self.skup_x, self.prefix, self.level)

    def fit2(self, stablo, skup_x, prefix, level):
        if not stablo:
            return None

        visited_clan = {}
        for clan in stablo:
            member = clan[-1]
            if member in visited_clan:
                visited_clan[member] += 1
            else:
                visited_clan[member] = 1

        if len(visited_clan) == 1 or (self.max_depth is not None and level > self.max_depth):
            leaf_label = max(sorted(visited_clan), key=visited_clan.get)
            print(f"{prefix} {leaf_label}")
            return leaf_label

        visited_labele = [count for count in visited_clan.values()]
        visited_counts = sum(visited_clan.values())

        entropy1 = E(visited_labele, visited_counts)

        vrati_najveci = []
        lista_clanova = []

        for index, x in enumerate(skup_x):
            lista = []

            for clan in stablo:
                if index < len(clan):
                    if clan[index] not in lista:
                        lista.append(clan[index])
            lista_dicta = []
            for clan1 in lista:
                clan_dict = {}
                for neki_znak in visited_clan:
                    clan_dict[neki_znak] = 0
                for znak in stablo:
                    if clan1 == znak[index]:
                        clan_dict[znak[-1]] += 1
                lista_dicta.append(clan_dict)
            
            lista_clanova.append(lista)
            vrati_najveci.append(IG(entropy1, lista_dicta, visited_counts))
        
        najveci_clan = max(vrati_najveci)
        index_najveceg_clana = vrati_najveci.index(najveci_clan)
        
        ime_glavnog = skup_x[index_najveceg_clana]

        novi_skup_x = skup_x.copy()
        novi_skup_x.pop(index_najveceg_clana)   


        subtree = {}
        for x in lista_clanova[index_najveceg_clana]:
            novo_stablo = []
            for red in stablo:
                if red[index_najveceg_clana] == x:
                    novi_red = red[:index_najveceg_clana] + red[index_najveceg_clana+1:]
                    novo_stablo.append(novi_red)

            novi_prefix = f"{prefix} {level}:{ime_glavnog}={x}"
            subtree[x] = self.fit2(novo_stablo, novi_skup_x, novi_prefix, level + 1)

        result = {ime_glavnog: subtree}

        return result

    def predict(self, example):
        return self.predict2(example, self.tree)

    def predict2(self, example, tree):
        if not isinstance(tree, dict):
            return tree

        root = list(tree.keys())[0]
        subtree = tree[root]
        feature_value = example[root]

        if feature_value in subtree:
            return self.predict2(example, subtree[feature_value])
        else:
            return None

def E(labela, counter):
    entropy = 0
    for clan in labela:
        if clan == 0:
            return 0
        entropy -= clan / counter * log(clan / counter, 2)
    return entropy

def IG(entropy, lista_listi, counter):
    for lista in lista_listi:
        lista_countova = []
        zbroj = 0
        for clan in lista:
            vrijednost = lista[clan]
            zbroj += vrijednost
            lista_countova.append(vrijednost)
        entropy -= zbroj / counter * E(lista_countova, zbroj)
    return entropy

def main():

    citaj = args[1]
    citaj2 = args[2]

    max_depth = None

    if len(args) >= 4 :
        max_depth = int(args[3])
        
    datoteka = CitajIzDatoteke(citaj)
    tabela = datoteka[1:]
    stablo = []

    linija = datoteka[0].strip().split(",")

    set_xeva = []
    for nesto in linija[:-1]:
        set_xeva.append(nesto)

    for line in tabela:
        line = line.strip()
        line1 = line.split(",")
        stablo.append(line1)
    print("[BRANCHES]:")
    User = ID3(stablo, set_xeva, linija[-1], max_depth=max_depth)
    User.fit()

    testna_datoteka = CitajIzDatoteke(citaj2)
    testna_tabela = [line.strip().split(",") for line in testna_datoteka[1:]]

    predictions = []
    correct = 0
    actual_labels = []
    unique_labels = set()
    train_target_values = [row[-1] for row in stablo]
    most_common_train_target = max(sorted(set(train_target_values)), key=train_target_values.count)

    for example in testna_tabela:
        example_dict = {}
        for i in range(len(example) - 1):
            example_dict[linija[i]] = example[i]
        true_label = example[-1]
        actual_labels.append(true_label)
        unique_labels.add(true_label)
        prediction = User.predict(example_dict)

        if prediction is None:
            prediction = most_common_train_target

        predictions.append(prediction)
        if prediction == true_label:
            correct += 1

    total = len(testna_tabela)
    accuracy = correct / total

 
    unique_labels = sorted(unique_labels)
    label_index = {}
    for i, label in enumerate(unique_labels):
        label_index[label] = i
    matrix_size = len(unique_labels)
    confusion_matrix = []
    for _ in range(matrix_size):
        row = [0] * matrix_size
        confusion_matrix.append(row)

    for i in range(len(actual_labels)):
        actual = actual_labels[i]
        predicted = predictions[i]
        if predicted is not None:
            actual_index = label_index.get(actual)
            predicted_index = label_index.get(predicted)
            if actual_index is not None and predicted_index is not None:
                confusion_matrix[actual_index][predicted_index] += 1


    print("[PREDICTIONS]:", " ".join(predictions))
    print(f"[ACCURACY]: {accuracy:.5f}")
    print("[CONFUSION_MATRIX]:")
    for row in confusion_matrix:
        print(" ".join(map(str, row)))

if __name__ == "__main__":
    main()
