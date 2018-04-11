## Juan Carlos Tapia Flores
## 14133

import string

## Diccionario con frequencia de palabras
def listaPalabrasDicFrec(lista):
    cadenaPalabras = ""
    for oracion in lista:
        cadenaPalabras += oracion + " "
    listaPalabras = cadenaPalabras.split()

    frecs = dict([])
    for p in listaPalabras:
        if p in frecs.keys():
            frecs[p] += 1
        else:
            frecs[p] = 1
                    
    return frecs

## Frecuencia de una palabra en la lista
def frecuenciaEnLista(frec_list, word):
    if not word in frec_list.keys():
        return 0
    else:
        return frec_list[word]


#Palabras diferentes
def palabrasTotales(spam_frec, ham_frec):
    return len(spam_frec) + len(list(set(ham_frec) - set(spam_frec)))

def totalFrecquencias(frecs):
    cont = 0
    for frec in frecs:
        cont += frecs[frec]
    return cont

## Borrar caracteres
def limpiar(oracion):
    return oracion.translate(None, string.punctuation).replace("\"", "").replace("\n", "").lower()


class Clasificador:
    k = 1
    spam_list = []
    ham_list = []
    spam_frec = []
    ham_frec = []
    X_palabras = 0
    total_frequencias_spam = 0
    total_frequencias_ham = 0
    
    def __init__(self, data):
        self.k = 1
        for line in data:
            l = line.split("\t")
            tipo = l[0]
            mensaje = limpiar(l[1])
            if tipo == "spam":
                self.spam_list.append(mensaje)
            else:
                self.ham_list.append(mensaje)

        self.spam_frec =  listaPalabrasDicFrec(self.spam_list)
        self.ham_frec =  listaPalabrasDicFrec(self.ham_list)
        self.X_palabras =  palabrasTotales(self.spam_frec, self.ham_frec)
        self.total_frequencias_spam = totalFrecquencias(self.spam_frec)
        self.total_frequencias_ham = totalFrecquencias(self.ham_frec)

            
    def p_spam (self):
        return float(len(self.spam_list) + self.k) / ( len(self.spam_list) + len(self.ham_list)+ self.k * 2)

    def p_ham (self):
        return float(len(self.ham_list) + self.k) / ( len(self.spam_list) + len(self.ham_list)+ self.k * 2)

    # P(word | spam)
    def p_word_given_something(self, word, spam_or_ham):
        if (spam_or_ham == "spam") :
            used = self.spam_frec
            f_used = self.total_frequencias_spam
        else:
            used = self.ham_frec
            f_used = self.total_frequencias_ham
        return float(frecuenciaEnLista(used, word) + self.k) / (f_used + self.k * self.X_palabras ) 


    # P(spam | sentence)
    def p_X_given_sentence(self, sentence):
        sentenceC = limpiar(sentence)

        words = sentenceC.split()

        used = "spam"
        not_used = "ham"
        
        p1 = 1
        for word in words:
            p1 *= self.p_word_given_something(word, used)
        p1 *= self.p_spam()

        p2 = 1
        for word in words:
            p2 *= self.p_word_given_something(word, not_used)
        p2 *= self.p_ham()
    
        if (p1== 0  and p2 == 0):
            return 0
        return p1/(p1+p2)


    def change_k(self, k):
        self.k = k



def cross_validation(cross_data, classif):
    lines = []
    for line in cross_data:
        l = line.split("\t")
        tipo = l[0]
        mensaje = limpiar(l[1])
        lines.append((mensaje, tipo))

    true_types = dict(lines)
    

    success_rate = 0
    k = 0
    for i in range (0, 10000):
        correct_count = 0
        error_count = 0
        k_i = 0.001* (i+1)
        classif.change_k(k_i)
        for sentence in true_types:
            p = classif.p_X_given_sentence(sentence)
            

            if p < 0.5:
                r = "ham"
            else:
                r = "spam"

            if true_types[sentence] == r:
                correct_count +=1
            else:
                error_count += 1
            
        success_rate_i = float(correct_count) / len(true_types)

        if (success_rate_i > success_rate):
            success_rate = success_rate_i
            k = k_i
        print str(i) +"  -  k = "+str(k_i) +"  -  "+ str(success_rate_i) 

    return k



def printResults(test_list):
    fileR = open("results.txt", "w")
    for line in test_list:
        p = classif.p_X_given_sentence(line)
            
        if p < 0.5:
            r = "ham"
        else:
            r = "spam"

        fileR.write(r+"\t"+line)
    fileR.close() 

    
file = open("corpus.txt", "r")
points = []

spam_list = []
ham_list = []



data = []
for line in file:
    data.append(line)



          
cont = 0
train = round(len(data)*0.8) 
cross = train + round(len(data)*0.1)

train_data = []
cross_data = []
test_data = []
for line in data:
    if cont < train:
        train_data.append(line)        
    elif cont < cross:
        cross_data.append(line)
    else:
        test_data.append(line.split("\t")[1])
    cont += 1 




classif = Clasificador(train_data)
print "Training Complete"


#print cross_validation(cross_data, classif)
file = open("test_sms.txt", "r")
test_data = []
for line in file:
    test_data.append(line)
    
printResults(test_data)


file.close()
