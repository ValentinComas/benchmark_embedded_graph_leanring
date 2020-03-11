def featToArray(dataset_name):
    f = open(dataset_name + ".feat", "r")
    lines_list = f.read().splitlines()
    feat_lists = []
    for i in range(len(lines_list)):
        feat_lists.append(lines_list[i].split(' '))
    return feat_lists

def save_order(arrayToSave,path):
    mydict = {}
    j = 0
    for i in arrayToSave:
        mydict[i] = j
        j+=1
        table = []
    for i in range(1,len(mydict)+1):
        table.append(str(mydict[str(i)]))
    f = open(path, 'w')
    for i in table:
        f.write(i + '\n')
    f.close()