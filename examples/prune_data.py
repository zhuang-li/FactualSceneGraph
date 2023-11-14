line_set = set()
for line in open(file='spice_input.txt').readlines():
    line_set.add(line.strip().replace('p:','').replace(':1','').replace(':2','').replace(':3','').replace(':4',''))

fp = open(file='prune_spice_output.txt', mode='w')
for line in line_set:
    fp.write(line+'\n')