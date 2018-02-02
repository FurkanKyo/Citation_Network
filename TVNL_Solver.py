from snapvx import *
import time
import numpy as np
import networkx as nx
import os
import json
from random import shuffle

DEFAULT = object()

def laplace_reg(src, dst, data):
    return (lambda_s_e*sum_squares(src['x'] - dst['x']), [])

# Test performance via total loss on train set
def train_loss_calculator(gvx, num_nodes, x_dimension, train_per_node):
    train_loss = 0
    # T1
    for i in range(num_nodes):
        if i <= 2:
            x = gvx.GetNodeValue(i + 1, 'x')
            for k in range(train_per_node):
                w = w1[i * train_per_node + k]
                noise = noise1[i * train_per_node + k]
                #             w = np.random.normal(0, 1, 50)
                #             noise = np.random.normal(0, 0.1, 1)
                y = np.dot(a1[:x_dimension], w) + a1[x_dimension] + noise
                # print(y)
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                train_loss += square(pred_y - y)

        else:
            x = gvx.GetNodeValue(i + 1, 'x')

            for k in range(train_per_node):
                w = w1[i * train_per_node + k]
                noise = noise1[i * train_per_node + k]
                #             w = np.random.normal(0, 1, 50)
                #             noise = np.random.normal(0, 0.1, 1)
                y = np.dot(a2[:x_dimension], w) + a2[x_dimension] + noise
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                train_loss += square(pred_y - y)
    # T2
    for i in range(num_nodes):
        if i < 2:
            x = gvx.GetNodeValue(5 + i + 1, 'x')

            for k in range(train_per_node):
                w = w2[i * train_per_node + k]
                noise = noise2[i * train_per_node + k]
                #             w = np.random.normal(0, 1, 50)
                #             noise = np.random.normal(0, 0.1, 1)
                y = np.dot(a1_2[:x_dimension], w) + a1_2[x_dimension] + noise
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                train_loss += square(pred_y - y)

        else:
            x = gvx.GetNodeValue(5 + i + 1, 'x')

            for k in range(train_per_node):
                w = w2[i * train_per_node + k]
                noise = noise2[i * train_per_node + k]
                #             w = np.random.normal(0, 1, 50)
                #             noise = np.random.normal(0, 0.1, 1)
                y = np.dot(a2_2[:x_dimension], w) + a2_2[x_dimension] + noise
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                train_loss += square(pred_y - y)
    # T3
    for i in range(num_nodes):
        if i <= 2:
            x = gvx.GetNodeValue(10 + i + 1, 'x')

            for k in range(train_per_node):
                w = w3[i * train_per_node + k]
                noise = noise3[i * train_per_node + k]
                #             w = np.random.normal(0, 1, 50)
                #             noise = np.random.normal(0, 0.1, 1)
                y = np.dot(a1_3[:x_dimension], w) + a1_3[x_dimension] + noise
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                train_loss += square(pred_y - y)

        else:
            x = gvx.GetNodeValue(10 + i + 1, 'x')

            for k in range(train_per_node):
                w = w3[i * train_per_node + k]
                noise = noise3[i * train_per_node + k]
                #             w = np.random.normal(0, 1, 50)
                #             noise = np.random.normal(0, 0.1, 1)
                y = np.dot(a2_3[:x_dimension], w) + a2_3[x_dimension] + noise
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                train_loss += square(pred_y - y)

    print('Train Loss: ', train_loss.value)


# Test performance via total loss on random test set
def test_loss_calculator(gvx, num_nodes, x_dimension, test_per_node):
    test_loss = 0
    np.random.seed(3)
    cos_sim_t1 = 0
    cos_sim_t2 = 0
    cos_sim_t3 = 0
    # T1
    for i in range(num_nodes):
        if i <= 2:
            x = gvx.GetNodeValue(i+1,'x')
            cos_sim_t1 += np.dot(x,a1) / (sqrt(sum([a*a for a in x]))*sqrt(sum([a*a for a in a1])))
            for k in range(test_per_node):
                w = np.random.normal(0, 1, n-1)
                noise = np.random.normal(0, reg_noise, 1)
                y =  np.dot(a1[:x_dimension], w) + a1[x_dimension]
                #print(y)
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                test_loss += square(pred_y - y)

        else:
            x = gvx.GetNodeValue(i+1,'x')
            cos_sim_t1 += np.dot(x,a2) / (sqrt(sum([a*a for a in x]))*sqrt(sum([a*a for a in a2])))
            for k in range(test_per_node):
                w = np.random.normal(0, 1, n-1)
                noise = np.random.normal(0, reg_noise, 1)
                y =  np.dot(a2[:x_dimension], w) + a2[x_dimension]
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                test_loss += square(pred_y - y)
    # T2
    for i in range(num_nodes):
        if i < 2:
            x = gvx.GetNodeValue(5+i+1,'x')
            cos_sim_t2 += np.dot(x,a1_2) / (sqrt(sum([a*a for a in x]))*sqrt(sum([a*a for a in a1_2])))
            for k in range(test_per_node):
                w = np.random.normal(0, 1, n-1)
                noise = np.random.normal(0, reg_noise, 1)
                y =  np.dot(a1_2[:x_dimension], w) + a1_2[x_dimension]
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                test_loss += square(pred_y - y)

        else:
            x = gvx.GetNodeValue(5+i+1,'x')
            cos_sim_t2 += np.dot(x,a2_2) / (sqrt(sum([a*a for a in x]))*sqrt(sum([a*a for a in a2_2])))
            for k in range(test_per_node):
                w = np.random.normal(0, 1, n-1)
                noise = np.random.normal(0, reg_noise, 1)
                y =  np.dot(a2_2[:x_dimension], w) + a2_2[x_dimension]
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                test_loss += square(pred_y - y)
    # T3
    for i in range(num_nodes):
        if i <= 2:
            x = gvx.GetNodeValue(10+i+1,'x')
            cos_sim_t3 += np.dot(x,a1_3) / (sqrt(sum([a*a for a in x]))*sqrt(sum([a*a for a in a1_3])))
            for k in range(test_per_node):
                w = np.random.normal(0, 1, n-1)
                noise = np.random.normal(0, reg_noise, 1)
                y =  np.dot(a1_3[:x_dimension], w) + a1_3[x_dimension]
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                test_loss += square(pred_y - y)

        else:
            x = gvx.GetNodeValue(10+i+1,'x')
            cos_sim_t3 += np.dot(x,a2_3) / (sqrt(sum([a*a for a in x]))*sqrt(sum([a*a for a in a2_3])))
            for k in range(test_per_node):
                w = np.random.normal(0, 1, n-1)
                noise = np.random.normal(0, reg_noise, 1)
                y =  np.dot(a2_3[:x_dimension], w) + a2_3[x_dimension]
                pred_y = np.dot(x[:x_dimension], w) + x[x_dimension]
                test_loss += square(pred_y - y)

    print('Test loss', test_loss.value)
    print('Similarities: ', cos_sim_t1.value, cos_sim_t2.value, cos_sim_t3.value)



def graph_optimizer(total_steps, graph_path, snapshots_path, G_features_path, lambda_f, lambda_f_r, lambda_s_e, lambda_t_e, num_of_snapshots = DEFAULT):

    #Read the graph
    G = nx.read_gpickle(graph_path)
    print(nx.info(G))
    num_nodes = len(nx.nodes(G))
    #Read the snapshots -- file names are sorted in the same order as snapshots
    if num_of_snapshots == DEFAULT:
        num_of_snapshots = len(os.listdir(snapshots_path))
    else:
        num_of_snapshots = num_of_snapshots

    print('Num of snapshots: ', num_of_snapshots)

    list_of_snapshot_dirs = []
    for dir in os.listdir(snapshots_path):
        with open(snapshots_path + '/' + dir) as file:
            snapshot_dir = json.load(file)
            list_of_snapshot_dirs.append(snapshot_dir)


    # Read features dict

    features_dict = {}
    with open(G_features_path) as file:
        features_dict = json.load(file)

    if isinstance(features_dict[list(features_dict.keys())[0]][0], list):
        num_features = len(features_dict[list(features_dict.keys())[0]][0])
    else:
        num_features = len(features_dict[list(features_dict.keys())[0]])
    print('Num of features on each node: ', num_features)

    print('Testing node counts:', len(features_dict.keys()) == num_nodes, len(features_dict.keys()), num_nodes)

    for step in range(total_steps):

        gvx = TGraphVX()

        node_id_key_dict = {}
        node_key_id_dict = {}
        node_index = 0
        for s in range(num_of_snapshots):

            t_0 = time.time()

            for key in features_dict.keys():

                x = Variable(num_features, name='x')
                #beta = Variable(num_features, name='b')
                obj = 0
                pred_y = 0
                if isinstance(features_dict[key][0], list):
                    for i in range(10): # len(features_dict[key])
                        pred_y = 0

                        w = features_dict[key][i]

                        y = list_of_snapshot_dirs[s][key][i]

                        for l in range(num_features):
                            pred_y += x[l] * w[l]
                        #pred_y += x[-1] # Add bias
                        obj += lambda_f*square(y - pred_y)

                else:
                    w = features_dict[key]

                    y = list_of_snapshot_dirs[s][key]

                    for l in range(num_features):
                        pred_y += x[l] * w[l]
                    #pred_y += x[-1]  # Add bias
                    obj += lambda_f * square(y - pred_y)

                obj += lambda_f_r * sum(x) # Add regression regularizor
                #obj += lambda_f_r * sum_squares(beta)
                # Add the key node
                constraints = [x >= 0]
                gvx.AddNode(node_index, Objective = obj, Constraints=constraints)

                node_id_key_dict[node_index] = key

                if key in node_key_id_dict.keys():
                    node_key_id_dict[key].append(node_index)
                else:
                    node_key_id_dict[key] = [node_index]

                node_index += 1
            print('Nodes are added for snapshot ', s, ' in ', time.time()-t_0, ' seconds')
            # Add static edges in the snapshot s

            edge_count = 0
            #print(nx.edges(G))
            for edge in nx.edges(G):
                edge_count += 1
                #if edge_count % 10 == 0:
                src = gvx.GetNodeVariables(node_key_id_dict[edge[0]][s])
                dst = gvx.GetNodeVariables(node_key_id_dict[edge[1]][s])
                gvx.AddEdge(node_key_id_dict[edge[0]][s], node_key_id_dict[edge[1]][s], Objective = lambda_s_e*sum_squares(src['x'] - dst['x'] ))

            print('End of snapshot, ', s, ' in ', time.time()-t_0, ' seconds')

        # Add temporal edges
        for key in node_key_id_dict.keys():
            value = node_key_id_dict[key]
            for i in range(len(value)):
                if i < len(value)-1:
                    src = gvx.GetNodeVariables(value[i])
                    dst = gvx.GetNodeVariables(value[i+1])
                    gvx.AddEdge(value[i], value[i+1], Objective= lambda_t_e*sum_squares(src['x'] - dst['x']))
        print('All temporal edges added!')

        # Add edge objectives
        #gvx.AddEdgeObjectives(laplace_reg)

        print('Test total node count, ', node_index == num_of_snapshots*num_nodes, node_index, num_of_snapshots*num_nodes)

        # Graph is ready, now solve it!!
        print('Step :', step, ' ***')
        t_0 = time.time()
        gvx.Solve(UseADMM = True, Verbose=True) #UseADMM=True
        t_1 = time.time()

        print('Total time passed to solve: ', str(t_1-t_0))
        print('Solution: ', gvx.value)
        for i in range(node_index):
            print(node_id_key_dict[i], gvx.GetNodeValue(i, 'x'))
            #print(node_id_key_dict[i], gvx.GetNodeValue(i, 'b'))
            # for s in range(num_of_snapshots):
            #     if isinstance(features_dict[key], list):
            #         test_obj = 0
            #         for j in range(1):  # len(features_dict[key])
            #             test_pred_y = 0
            #             test_w = features_dict[node_id_key_dict[i]][j]
            #             test_y = list_of_snapshot_dirs[s][node_id_key_dict[i]][j]
            #             for l in range(len(test_w)):
            #                 test_pred_y += gvx.GetNodeValue(i, 'x')[l] * test_w[l]
            #             test_pred_y += gvx.GetNodeValue(i, 'x')[-1]  # Add bias
            #             test_obj += lambda_f * (test_pred_y - test_y)*(test_pred_y - test_y)
            #             print(test_w)
            #             print(test_y, test_pred_y)
            #             print(test_obj)


# Synthetic Example
def graph_optimizer_synthetic(total_steps, num_nodes, n, m, train_per_node, test_per_node, x_dimension, reg_noise):
    for step in range(total_steps):
        if step == 0:
            saved_variables = False
        else:
            saved_variables = True

        gvx = TGraphVX()
        # T1
        # For each node, add an objective
        for i in range(num_nodes):
            if i <= 2:
                x = Variable(n, name='x')
                obj = 0
                for k in range(train_per_node):
                    if saved_variables:
                        w = w1[i * train_per_node + k]  # np.random.normal(0, 1, 50) #
                        # w1.append(w)
                        noise = noise1[i * train_per_node + k]  # np.random.normal(0, 0.1, 1) #
                        # noise1.append(noise)
                    else:
                        w = np.random.normal(0, 1, n - 1)  # w1[i*25+k] #
                        w1.append(w)
                        noise = np.random.normal(0, reg_noise, 1)  # noise1[i*25+k] #
                        noise1.append(noise)

                    y = (np.dot(a1[:x_dimension], w) + a1[x_dimension] + noise)  # [0]
                    pred_y = 0
                    for l in range(x_dimension):
                        pred_y += x[l] * w[l]
                    pred_y += x[x_dimension]
                    obj += square(pred_y - y)

                    #         cons = [-1 <= val for val in x]
                    #         cons = cons + [1 >= val for val in x]
                obj += u_reg * sum_squares(x[:x_dimension])
                gvx.AddNode(i + 1, obj, cons)
            else:
                x = Variable(n, name='x')
                obj = 0
                for k in range(train_per_node):
                    if saved_variables:
                        w = w1[i * train_per_node + k]  # np.random.normal(0, 1, 50) #
                        # w1.append(w)
                        noise = noise1[i * train_per_node + k]  # np.random.normal(0, 0.1, 1) #
                        # noise1.append(noise)
                    else:
                        w = np.random.normal(0, 1, n - 1)  # w1[i*25+k] #
                        w1.append(w)
                        noise = np.random.normal(0, reg_noise, 1)  # noise1[i*25+k]#
                        noise1.append(noise)

                    y = (np.dot(a2[:x_dimension], w) + a2[x_dimension] + noise)  # [0]
                    pred_y = 0
                    for l in range(x_dimension):
                        pred_y += x[l] * w[l]
                    pred_y += x[x_dimension]
                    obj += square(pred_y - y)

                    #         cons = [-1 <= val for val in x]
                    #         cons = cons + [1 >= val for val in x]
                obj += u_reg * sum_squares(x[:x_dimension])
                gvx.AddNode(i + 1, obj, cons)
        gvx.AddEdge(1, 2)
        gvx.AddEdge(1, 3)
        gvx.AddEdge(2, 3)
        gvx.AddEdge(3, 4)
        gvx.AddEdge(4, 5)

        # T2
        # For each node, add an objective
        for i in range(num_nodes):
            if i < 2:
                x = Variable(n, name='x')
                obj = 0
                for k in range(train_per_node):
                    if saved_variables:
                        w = w2[i * train_per_node + k]  # np.random.normal(0, 1, 50) #
                        # w2.append(w)
                        noise = noise2[i * train_per_node + k]  # np.random.normal(0, 0.1, 1)#
                        # noise2.append(noise)
                    else:
                        w = np.random.normal(0, 1, n - 1)  # w2[i*25+k] #
                        w2.append(w)
                        noise = np.random.normal(0, reg_noise, 1)  # noise2[i*25+k] #
                        noise2.append(noise)

                    y = (np.dot(a1_2[:x_dimension], w) + a1_2[x_dimension] + noise)  # [0]
                    pred_y = 0
                    for l in range(x_dimension):
                        pred_y += x[l] * w[l]
                    pred_y += x[x_dimension]
                    obj += square(pred_y - y)

                    #         cons = [-1 <= val for val in x]
                    #         cons = cons + [1 >= val for val in x]
                obj += u_reg * sum_squares(x[:x_dimension])
                gvx.AddNode(5 + i + 1, obj, cons)
            else:
                x = Variable(n, name='x')
                obj = 0
                for k in range(train_per_node):
                    if saved_variables:
                        w = w2[i * train_per_node + k]  # np.random.normal(0, 1, 50) #
                        # w2.append(w)
                        noise = noise2[i * train_per_node + k]  # np.random.normal(0, 0.1, 1) #
                        # noise2.append(noise)
                    else:
                        w = np.random.normal(0, 1, n - 1)  # w2[i*25+k] #
                        w2.append(w)
                        noise = np.random.normal(0, reg_noise, 1)  # noise2[i*25+k] #
                        noise2.append(noise)

                    y = (np.dot(a2_2[:x_dimension], w) + a2_2[x_dimension] + noise)  # [0]
                    pred_y = 0
                    for l in range(x_dimension):
                        pred_y += x[l] * w[l]
                    pred_y += x[x_dimension]
                    obj += square(pred_y - y)

                    #         cons = [-1 <= val for val in x]
                    #         cons = cons + [1 >= val for val in x]
                obj += u_reg * sum_squares(x[:x_dimension])
                gvx.AddNode(5 + i + 1, obj, cons)
        gvx.AddEdge(6, 7)
        gvx.AddEdge(6, 8)
        gvx.AddEdge(7, 8)
        gvx.AddEdge(8, 9)
        gvx.AddEdge(9, 10)

        # T3
        # For each node, add an objective
        for i in range(num_nodes):
            if i <= 2:
                x = Variable(n, name='x')
                obj = 0
                for k in range(train_per_node):
                    if saved_variables:
                        w = w3[i * train_per_node + k]  # np.random.normal(0, 1, 50) #
                        # w3.append(w)
                        noise = noise3[i * train_per_node + k]  # np.random.normal(0, 0.1, 1) #
                        # noise3.append(noise)
                    else:
                        w = np.random.normal(0, 1, n - 1)  # w3[i*25+k] #
                        w3.append(w)
                        noise = np.random.normal(0, reg_noise, 1)  # noise3[i*25+k] #
                        noise3.append(noise)

                    y = (np.dot(a1_3[:x_dimension], w) + a1_3[x_dimension] + noise)  # [0]
                    pred_y = 0
                    for l in range(x_dimension):
                        pred_y += x[l] * w[l]
                    pred_y += x[x_dimension]
                    obj += square(pred_y - y)

                    #         cons = [-1 <= val for val in x]
                    #         cons = cons + [1 >= val for val in x]
                obj += u_reg * sum_squares(x[:x_dimension])
                gvx.AddNode(10 + i + 1, obj, cons)
            else:
                x = Variable(n, name='x')
                obj = 0
                for k in range(train_per_node):
                    if saved_variables:
                        w = w3[i * train_per_node + k]  # np.random.normal(0, 1, 50) #
                        # w3.append(w)
                        noise = noise3[i * train_per_node + k]  # np.random.normal(0, 0.1, 1) #
                        # noise3.append(noise)
                    else:
                        w = np.random.normal(0, 1, n - 1)  # w3[i*25+k] #
                        w3.append(w)
                        noise = np.random.normal(0, reg_noise, 1)  # noise3[i*25+k] #
                        noise3.append(noise)

                    y = (np.dot(a2_3[:x_dimension], w) + a2_3[x_dimension] + noise)  # [0]
                    pred_y = 0
                    for l in range(x_dimension):
                        pred_y += x[l] * w[l]
                    pred_y += x[x_dimension]
                    obj += square(pred_y - y)

                    #         cons = [-1 <= val for val in x]
                    #         cons = cons + [1 >= val for val in x]
                obj += u_reg * sum_squares(x[:x_dimension])
                gvx.AddNode(10 + i + 1, obj, cons)
        gvx.AddEdge(11, 12)
        gvx.AddEdge(11, 13)
        gvx.AddEdge(12, 13)
        gvx.AddEdge(13, 14)
        gvx.AddEdge(14, 15)

        gvx.AddEdgeObjectives(laplace_reg)

        if step == 0:

            gvx.AddEdge(1, 6,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(1)['x'] - gvx.GetNodeVariables(6)['x']),
                        Constraints=[])
            gvx.AddEdge(2, 7,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(2)['x'] - gvx.GetNodeVariables(7)['x']),
                        Constraints=[])
            gvx.AddEdge(3, 8,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(3)['x'] - gvx.GetNodeVariables(8)['x']),
                        Constraints=[])
            gvx.AddEdge(4, 9,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(4)['x'] - gvx.GetNodeVariables(9)['x']),
                        Constraints=[])
            gvx.AddEdge(5, 10,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(5)['x'] - gvx.GetNodeVariables(10)['x']),
                        Constraints=[])
            gvx.AddEdge(6, 11,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(6)['x'] - gvx.GetNodeVariables(11)['x']),
                        Constraints=[])
            gvx.AddEdge(7, 12,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(7)['x'] - gvx.GetNodeVariables(12)['x']),
                        Constraints=[])
            gvx.AddEdge(8, 13,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(8)['x'] - gvx.GetNodeVariables(13)['x']),
                        Constraints=[])
            gvx.AddEdge(9, 14,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(9)['x'] - gvx.GetNodeVariables(14)['x']),
                        Constraints=[])
            gvx.AddEdge(10, 15,
                        Objective=e_lambda * sum_squares(gvx.GetNodeVariables(10)['x'] - gvx.GetNodeVariables(15)['x']),
                        Constraints=[])

        else:
            gvx.AddEdge(1, 6, Objective=e_lambda * sum_squares(
                Betas.value[:, 0] + gvx.GetNodeVariables(1)['x'] - gvx.GetNodeVariables(6)['x']), Constraints=[])
            gvx.AddEdge(2, 7, Objective=e_lambda * sum_squares(
                Betas.value[:, 1] + gvx.GetNodeVariables(2)['x'] - gvx.GetNodeVariables(7)['x']), Constraints=[])
            gvx.AddEdge(3, 8, Objective=e_lambda * sum_squares(
                Betas.value[:, 2] + gvx.GetNodeVariables(3)['x'] - gvx.GetNodeVariables(8)['x']), Constraints=[])
            gvx.AddEdge(4, 9, Objective=e_lambda * sum_squares(
                Betas.value[:, 3] + gvx.GetNodeVariables(4)['x'] - gvx.GetNodeVariables(9)['x']), Constraints=[])
            gvx.AddEdge(5, 10, Objective=e_lambda * sum_squares(
                Betas.value[:, 4] + gvx.GetNodeVariables(5)['x'] - gvx.GetNodeVariables(10)['x']), Constraints=[])
            gvx.AddEdge(6, 11, Objective=e_lambda * sum_squares(
                Betas.value[:, 5] + gvx.GetNodeVariables(6)['x'] - gvx.GetNodeVariables(11)['x']), Constraints=[])
            gvx.AddEdge(7, 12, Objective=e_lambda * sum_squares(
                Betas.value[:, 6] + gvx.GetNodeVariables(7)['x'] - gvx.GetNodeVariables(12)['x']), Constraints=[])
            gvx.AddEdge(8, 13, Objective=e_lambda * sum_squares(
                Betas.value[:, 7] + gvx.GetNodeVariables(8)['x'] - gvx.GetNodeVariables(13)['x']), Constraints=[])
            gvx.AddEdge(9, 14, Objective=e_lambda * sum_squares(
                Betas.value[:, 8] + gvx.GetNodeVariables(9)['x'] - gvx.GetNodeVariables(14)['x']), Constraints=[])
            gvx.AddEdge(10, 15, Objective=e_lambda * sum_squares(
                Betas.value[:, 9] + gvx.GetNodeVariables(10)['x'] - gvx.GetNodeVariables(15)['x']), Constraints=[])

        # Graph is ready, now solve it!!
        print('Step :', step, ' ***')
        t_0 = time.time()
        gvx.Solve()
        t_1 = time.time()

        #     print('Total time passed: ', str(t_1-t_0))
        print('Solution: ', gvx.value)

        # Split loss into terms

        edge_penalties = 0
        for ei in gvx.Edges():
            src_id = ei.GetSrcNId()
            src_vars = gvx.GetNodeValue(src_id, 'x')
            dst_id = ei.GetDstNId()
            dst_vars = gvx.GetNodeValue(dst_id, 'x')
            edge_diff = src_vars - dst_vars
            edge_penalties += sum([x * x for x in edge_diff])

        B_term_1_T = 0
        B_term_1 = 0
        for i in range(n):
            penalties_B_X = gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(i + 1 + 5, 'x')
            if step == 0:
                penalties_B_X_T = penalties_B_X
            else:
                penalties_B_X_T = np.squeeze(np.asarray(Betas.value[:, i])) + gvx.GetNodeValue(i + 1,'x') - gvx.GetNodeValue(i + 1 + 5, 'x')
            B_term_1 += sum([x * x for x in penalties_B_X])
            B_term_1_T += e_lambda * sum([x * x for x in penalties_B_X_T])

        edge_penalties = edge_penalties - B_term_1
        print ('F(x)value: ', gvx.value - B_term_1_T - e_pen * edge_penalties)
        print ('Static edge penalties: ', e_pen * edge_penalties, edge_penalties)
        print ('Temporal edge penalties: ', B_term_1_T, B_term_1_T / e_lambda)
        train_loss_calculator(gvx, num_nodes, x_dimension, train_per_node)

        # Construct the problem.
        Betas = Variable(n, m)
        total_loss = 0
        for i in range(m):
            total_loss += lambda_1 * sum_squares(Betas[:, i] + gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(i + 1 + 5, 'x'))
        # for i in range(m / 2):
        #     total_loss += lambda_2 * sum_squares(Betas[:, i] - Betas[:, i + 5])
        for i in range(m):
            total_loss += lambda_3 * (sum_squares(Betas[:, i]))

        objective = Minimize(total_loss)
        constraints = []
        # for i in range(n):
        #     for j in range(m):
        #         constraints = constraints + [-1 <= Betas[i,j], Betas[i,j] <= 1]
        prob = Problem(objective, constraints)

        # The optimal objective is returned by prob.solve().
        result = prob.solve()
        # The optimal value for x is stored in x.value.

        print ('Total Beta loss: ', result)

        # Penalty for each term
        Beta_term_1 = 0
        for i in range(m):
            penalties_B_X = np.squeeze(np.asarray(Betas.value[:, i])) + gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(i + 1 + 5, 'x')
            #     print('**', gvx.GetNodeValue(i+1,'x').shape)
            #     print('**', type(gvx.GetNodeValue(i+1,'x')))
            #     print('**', np.squeeze(np.asarray(Betas.value[:,i])).shape)

            #     print('**', type(np.squeeze(np.asarray(Betas.value[:,i]))))
            #     print('*****', penalties_B_X.shape)
            Beta_term_1 += lambda_1 * sum([x * x for x in penalties_B_X])
            # B_term_1 += lambda_1*sum([x*x for x in (Betas.value[:,i] + gvx.GetNodeValue(i+1,'x') - gvx.GetNodeValue(i+1+5,'x'))])
        print('Beta term 1: ', Beta_term_1, Beta_term_1 / lambda_1)

        Beta_term_2 = 0
        # for i in range(m / 2):
        #     Beta_term_2 += lambda_2 * sum([x * x for x in np.squeeze(np.asarray(Betas.value[:, i])) - np.squeeze(
        #         np.asarray(Betas.value[:, i + 5]))])
        # print('Beta term 2: ', Beta_term_2, Beta_term_2 / lambda_2)

        Beta_term_3 = 0
        for i in range(m / 2):
            Beta_term_3 += lambda_3 * sum([x * x for x in np.squeeze(np.asarray(Betas.value[:, i]))]) + lambda_3 * sum(
                [x * x for x in np.squeeze(np.asarray(Betas.value[:, i + 5]))])
        print('Beta term 3:', Beta_term_3, Beta_term_3 / lambda_3)

        # TEST RESULTS
        test_loss_calculator(gvx, num_nodes, x_dimension, test_per_node)

        print('**** End of step ', step, ' ****')


def beta_update(m, lambda_1, lambda_2, lambda_3):
    m = 2 * 5
    lambda_1 = 5
    lambda_2 = 5
    lambda_3 = 20

    # Construct the problem.
    Betas = Variable(n, m)
    total_loss = 0
    for i in range(m):
        total_loss += lambda_1 * sum_squares(
            Betas[:, i] + gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(i + 1 + 5, 'x'))
    for i in range(m / 2):
        total_loss += lambda_2 * sum_squares(Betas[:, i] - Betas[:, i + 5])
    for i in range(m / 2):
        total_loss += lambda_3 * (sum_squares(Betas[:, i]) + lambda_3 * sum_squares(Betas[:, i + 5]))

    objective = Minimize(total_loss)
    constraints = []
    # for i in range(n):
    #     for j in range(m):
    #         constraints = constraints + [-1 <= Betas[i,j], Betas[i,j] <= 1]
    prob = Problem(objective, constraints)

    # The optimal objective is returned by prob.solve().
    result = prob.solve()
    # The optimal value for x is stored in x.value.
    print Betas.value
    print result

    # Penalty for each term
    B_term_1 = 0
    for i in range(m):
        penalties_B_X = np.squeeze(np.asarray(Betas.value[:, i])) + gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(
            i + 1 + 5, 'x')
        #     print('**', gvx.GetNodeValue(i+1,'x').shape)
        #     print('**', type(gvx.GetNodeValue(i+1,'x')))
        #     print('**', np.squeeze(np.asarray(Betas.value[:,i])).shape)

        #     print('**', type(np.squeeze(np.asarray(Betas.value[:,i]))))
        #     print('*****', penalties_B_X.shape)
        B_term_1 += lambda_1 * sum([x * x for x in penalties_B_X])
        # B_term_1 += lambda_1*sum([x*x for x in (Betas.value[:,i] + gvx.GetNodeValue(i+1,'x') - gvx.GetNodeValue(i+1+5,'x'))])
    print B_term_1, B_term_1 / lambda_1

    B_term_2 = 0
    for i in range(m / 2):
        B_term_2 += lambda_2 * sum(
            [x * x for x in np.squeeze(np.asarray(Betas.value[:, i])) - np.squeeze(np.asarray(Betas.value[:, i + 5]))])
    print B_term_2, B_term_2 / lambda_2

    B_term_3 = 0
    for i in range(m / 2):
        B_term_3 += lambda_3 * sum([x * x for x in np.squeeze(np.asarray(Betas.value[:, i]))]) + lambda_3 * sum(
            [x * x for x in np.squeeze(np.asarray(Betas.value[:, i + 5]))])
    print B_term_3, B_term_3 / lambda_3

    return Betas



def loss_calculator():
    regression_penalties = 0
    ## ToDo -- Penalty check

    edge_penalties = 0
    for ei in gvx.Edges():
        src_id = ei.GetSrcNId()
        src_vars = gvx.GetNodeValue(src_id, 'x')
        dst_id = ei.GetDstNId()
        dst_vars = gvx.GetNodeValue(dst_id, 'x')
        edge_diff = src_vars - dst_vars
        # if abs(src_id - dst_id) != 5:

        edge_penalties += sum([x * x for x in edge_diff])

    B_term_1_T = 0
    B_term_1 = 0
    for i in range(10):
        penalties_B_X = gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(i + 1 + 5, 'x')
        penalties_B_X_T = np.squeeze(np.asarray(Betas.value[:, i])) + gvx.GetNodeValue(i + 1, 'x') - gvx.GetNodeValue(
            i + 1 + 5, 'x')
        B_term_1 += sum([x * x for x in penalties_B_X])
        B_term_1_T += r2 * sum([x * x for x in penalties_B_X_T])

    edge_penalties = edge_penalties - B_term_1
    print ('F(x)value: ', gvx.value - B_term_1_T - edge_penalties)
    print ('Edge penalties: ', edge_penalties)
    print ('Temporal edge penalties: ', B_term_1_T)


# np.random.seed(2)
#
# w1 = []
# noise1 = []
# w2 = []
# noise2 = []
# w3 = []
# noise3 = []
#
# n = 10
# a1 = np.random.normal(0, 1, n)
# a2 = np.random.normal(0, 1, n)
# a1_2 = a1 + np.random.normal(0, 0.1, n)
# a2_2 = a2 + np.random.normal(0, 0.1, n)
# a1_3 = a1_2 + np.random.normal(0, 0.1, n)
# a2_3 = a2_2 + np.random.normal(0, 0.1, n)
#
# num_nodes = 5
# num_edges = 5
#
# cons = []
#
#
#
# x_dimension = 9  # Actually this means dimension is 10 (0-9)
# n = 10
# train_per_node = 10
# test_per_node = 20
# reg_noise = 0.1
#
# u_reg = 0.01
# e_pen = 0.02
# e_lambda = 0.2
# lambda_1 = 0.2
# lambda_2 = 0
# lambda_3 = 0.1
#
# m = 2 * 5  # For beta (# of beta variables)


total_steps = 1
#graph_path = '/home/furkan/Desktop/SNAPVX/Graph.gpickle'
#snapshots_path = '/home/furkan/Desktop/SNAPVX/Snapshots/'
#G_features_path = '/home/furkan/Desktop/SNAPVX/Node_Features_Dict'
SF_base_path = '/home/furkan/Desktop/TVNL/Datasets/SF/'
graph_path = SF_base_path + 'SF_Graph.gpickle'
snapshots_path = SF_base_path + 'Snapshots/'
G_features_path = SF_base_path + 'Feature/SF_features.txt'
lambda_f = 1
lambda_f_r = 2
lambda_s_e = 1
lambda_t_e = 0
graph_optimizer(total_steps, graph_path, snapshots_path, G_features_path, lambda_f, lambda_f_r, lambda_s_e, lambda_t_e, num_of_snapshots=1)

