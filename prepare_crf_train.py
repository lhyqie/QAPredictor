fp = open ('train');
lines = fp.readlines()
fp.close()

fp = open('crf_train','w')
for line in lines:
    if line == '\n':
        fp.write('\n')
    else:
        pieces = line.split()
        fp.write(pieces[0]+"_"+pieces[1] + " ");
fp.close()