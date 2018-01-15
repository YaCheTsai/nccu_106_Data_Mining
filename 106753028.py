import math
import csv
from operator import itemgetter

#Problem 1 ( Cosine Similarity )
def cosine_similarity(A,B):
    Multiply = 0
    ABS_A = 0
    ABS_B = 0
    for i in range(0, len(A)):
        Multiply = Multiply + A[i]*B[i]
        ABS_A = ABS_A + A[i]*A[i]
        ABS_B = ABS_B + B[i]*B[i]
    result_value = Multiply / math.sqrt(ABS_A*ABS_B)
    return result_value

#Problem 2 ( Grades )
def grades(file_path):
    file = open(file_path, 'r')
    csvCursor = csv.reader(file)
    personDF = []
    for row in csvCursor:
        personDF.append(row)
    dict_grades = dict(personDF)
    tuple_grades = tuple(personDF, file_path)
    return dict_grades, tuple_grades

def tuple(personDF, file_path):
    arr_tuple = sorted(personDF[1:len(personDF)], key=itemgetter(0,1,2))
    result = "{"
    for i in range(0,len(arr_tuple)):
        result = result + "("
        for j in range(0, len(arr_tuple[0])):
            result = result + arr_tuple[i][j]
            if(j != 2 ):
                result = result + ","
        result = result + ")"
        if(i != len(arr_tuple)-1):
            result = result + ", "
    result = result +"}"
    with open(file_path, "wb") as csv_file:
        writer = csv.writer(csv_file, delimiter=',')
        for line in arr_tuple:
            writer.writerow(line)
    return result

def dict(personDF):
    arr_dict = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(1, len(personDF)):
        j = int(personDF[i][2]) / 10
        arr_dict[j]= arr_dict[j] + 1 
    result = "{"
    for i in range(0, 10):
        if(arr_dict[i] == 0):
            continue
        result = result + str(i) + "0~" + str(i) + "9:" + str(arr_dict[i])
        if(i!=9):
            result = result +", "
    return result +"}"


#Problem 3 ( The valid of password )
def valid_password(passwords):
    result = []
    for word in passwords:
        letter = word
        num = 0
        sp = False
        sp_char = ["%","!","?","#","@","$"]
        for letter in word:
            if (letter.isdigit()):
                num+=1
            for char in sp_char:
                if (char in letter):
                    sp = True 
                    break           
        length = len(word)>=5 and len(word)<=10
        letters =set(word)
        mixed = any(letter.islower() for letter in letters) and any(letter.isupper() for letter in letters)
        if(num >= 2 and sp and length and mixed):
            result.append(word)    
    result_list = result    
    return result_list

if __name__ == "__main__":
    pro_1_value = cosine_similarity([1,2,3],[4,5,6])
    pro_2_dict, pro_2_tuple = grades('./example.csv')
    pro_3_list = valid_password(['Ab12!','AA1234!?','AbCdEfGh','12345AaBa!', '12Zz!?98Aa#@'])
    print pro_1_value
    print pro_2_dict
    print pro_2_tuple
    print pro_3_list
