import sys
import re
import tree as tt


def readNode(fp):
    while True:
        line = fp.readline()
        if not line:
            return 0
        if not line.strip(' \n'):
            continue
        res = re.split(':', line.strip(' \n\r'))
        res2 = re.split(' ', res[0].strip())
        break

    while True:
        line = fp.readline()
        if not line:
            print 'Unexpected end of file.'
            sys.exit()
        if not line.strip(' \n'):
            continue
        res3 = re.split(':', line.strip(' \n\r'))
        res4 = re.split(' ', res3[1].strip())
        break

    nClasses = int(res3[0][3:].strip())
    if len(res4) != nClasses:
        print 'Wrong syntax. Number of classes do not agree.'
        sys.exit()
    classes = [cls.strip() for cls in res4]

    node = {'id': res2[0], 'desc': res[0][len(res2[0]):].strip(), 'parent_id': res[1].strip(), 'classes': classes}
    return node


def load(txt_fn):
    hrchy = tt.Tree()
    with open(txt_fn, 'rb') as fp:
        while True:
            node = readNode(fp)
            if not node:
                break
            if node['parent_id'] == 'root':
                hrchy.create_node({'desc': node['desc'], 'classes': node['classes']}, node['id'])
            else:
                hrchy.create_node({'desc': node['desc'], 'classes': node['classes']}, node['id'], node['parent_id'])
    return hrchy
