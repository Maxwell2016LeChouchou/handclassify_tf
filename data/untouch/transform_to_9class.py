f = open('/data4/wjz/gesture/shengteng/data/label_all_07_14.txt','r')
lines = f.readlines()
f.close()

num_lines = len(lines)
f = open('/data4/wjz/gesture/shengteng/data/label_all_mod_07_14.txt','w')
for i in range(num_lines):
    cur_line = lines[i].strip()
    cur_splits = cur_line.split()
    label = cur_splits[5]
    if label == '0' or label == '1' or label == '2':
        line = '%s %s %s %s %s 0'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '3':
        line = '%s %s %s %s %s 1'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '4':
        line = '%s %s %s %s %s 2'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '10' or label == '11':
        line = '%s %s %s %s %s 3'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '13' or label == '15':
        line = '%s %s %s %s %s 4'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '19':
        line = '%s %s %s %s %s 5'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '21':
        line = '%s %s %s %s %s 6'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '26':
        line = '%s %s %s %s %s 7'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
    elif label == '27':
        line = '%s %s %s %s %s 8'%(cur_splits[0], cur_splits[1],cur_splits[2],cur_splits[3],cur_splits[4])
        f.write(line+'\n')
f.close()
