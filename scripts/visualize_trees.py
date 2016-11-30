import pydot

def old_create_tree(transitions, words):
    template_start = """
    digraph G {
        nodesep=0.4; //was 0.8
        ranksep=0.5;
    """

    template_end = """
    }
    """

    template = ""
    template += template_start

    buf = list(reversed(words))
    stack = []
    leaves = []
    for i, t in enumerate(transitions):
        if t == 0:
            stack.append((i+1,t))
            leaves.append(str(i+1))
            template += '{node[label = "%s"]; %s;}\n' % (str(buf.pop()), str(i+1))
        else:
            right = stack.pop()
            left = stack.pop()
            top = i + 1
            stack.append((top, (left, right)))
            template += "{} -> {};\n".format(top, left[0])
            template += "{} -> {};\n".format(top, right[0])

    template += "{rank=same; %s}" % ("; ".join(leaves))
    template += template_end
    print(template)
    return stack, template

def old_main():
    sentence = [1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 7, 1, 4]
    transitions = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    stack, dd = old_create_tree(transitions, sentence)
    graphs = pydot.graph_from_dot_data(dd)
    open('graph.png', 'wb').write(graphs[0].create_jpeg())
    print(stack)

def write_tree(sentence, transitions, output_file):
    open(output_file, 'wb').write(graphs[0].create_jpeg())

if __name__ == '__main__':
    old_main()

    sentence = [1, 1, 9, 1, 1, 1, 1, 1, 1, 1, 7, 1, 4]
    transitions = [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1]
    output_file = "graph.png"

    
    
