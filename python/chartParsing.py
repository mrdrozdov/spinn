"""
Artifical test for chart parsing
"""

def m_compute_compositions(state):
        """h, c = state
                                l = (h[:, :-1, :], c[:, :-1, :])
                                r = (h[:, 1:, :], c[:, 1:, :])
                                #l = state
                                #r = state
                                length = l[0].size(1)
                                l_hiddens = l[0].chunk(length, dim=1)
                                l_states = l[1].chunk(length, dim=1)
                                r_hiddens = r[0].chunk(length, dim=1)
                                r_states = r[1].chunk(length, dim=1)"""

        #print("length of list:", length, len(l_hiddens))

        l_hiddens = state[:-1]
        r_hiddens = state[1:]
        chart = []
        weights = []

        for col in range(length):
            l = l_hiddens[col]
            r = r_hiddens[col] #, r_states[col])
            chart.append([compose(l=l, r=r)])
            weights.append([1.0])
            # initialise the remaining chart cells
            for row in range(1, col+1):
                chart[col].append(None)
                weights[col].append(None)

        for col in range(length):
            for row in range(1, col+1):
                # at row k, there are k possible way to form each constituent.
                # we try all of them, and keep the results around together with
                # the corresponding scores
                constituents = []
                hiddens = []
                cells = []
                scores = []
                for constituent_number in range(0, row):
                    l = chart[col-row+constituent_number][constituent_number]
                    r = chart[col][row-1-constituent_number]
                    constituents.append(compose(l=l, r=r))
                    hiddens.append(constituents[-1][0])
                    cells.append(constituents[-1][1])
                    # Not weighting
                    #comp_weights = dot_nd(
                    #    query=self.comp_query.weight.squeeze(),
                    #    candidates=hiddens[-1])
                    #scores.append(comp_weights) #--> append(scalar)
                    scores.append(1)

                # we gumbel-softmax the weights, and use them as a weighting mechanism
                # that strongly prefers assigning probability mass to only one
                # possibility
                #weights[col][row] = gumbel_softmax(torch.stack(scores, dim=2))
                weights[col][row] = 1

                #h_new = torch.sum(torch.mul(weights[col][row].unsqueeze(2), torch.stack(hiddens, dim=3)), dim=-1)
                #c_new = torch.sum(torch.mul(weights[col][row].unsqueeze(2), torch.stack(cells, dim=3)), dim=-1)
                h_new = combine(cells)

                chart[col][row] = (h_new)#, c_new)

        print chart

        #print len(chart), len(chart[1]), len(chart[1][1])
        return chart[length-1][length-1][0], chart[length-1][length-1][1], weights



sentence  = ["A", "B", "C", "D", "E", "F", "G", "."]

# Compose :  [A, B] = (A) + (B) = (AB) 
# Combine : ((AB)C), (A(BC)) = (ABC)
# A + B = (AB)
# (AB) + C = ((AB)C)

def compose(l, r):
    return "(" + l + r + ")"

def combine(list_versions):
    return list_versions[0].replace("(","").replace(")","")

def compute_compositions(sent):

    length = len(sent) -1
    l_hiddens = sent[:-1]
    l_cells = sent[:-1]
    r_hiddens = sent[1:]
    r_cells = sent[1:]
    chart = []
    weights = []

    """
    layer_0 = []
    for i in range(len(sent)):
        layer_0.append((sent[i], sent[i]))
    chart = [layer_0]
    """
    
    chart = [sent] # list or tuple. w/e
    for row in range(1, len(sent)): 
        chart.append([])
        for col in range(len(sent) - row):
            chart[row].append(None)
    
    for row in range(1, len(sent)): # = len(l_hiddens)
        for col in range(len(sent) - row):
            versions = []
            for i in range(row):
                print row, col, chart[row-i-1][col], chart[i][row+col-i]
                versions.append(compose(chart[row-i-1][col], chart[i][row+col-i]))

            chart[row][col] = combine(versions)
        
    return chart










