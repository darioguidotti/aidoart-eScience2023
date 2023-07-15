import copy
import os.path

import onnx
import pynever.strategies.training as pyn_training
import pynever.strategies.conversion as pyn_conversion
import pynever.datasets as pyn_data
import pynever.networks as pyn_net
import pynever.nodes as pyn_nodes
import numpy as np
import torch.optim as topt
import torch.nn as nn
import logging
import utilities
import torchvision.transforms as transforms

from datetime import datetime


# ===== SET EXPERIMENT ID AND FOLDERS CREATION =====
#
#
#
benchmark_datetime = datetime.now().strftime('%d-%m-%Y_%H:%M:%S')
experiment_folder = f"benchmarks_{benchmark_datetime}/"
onnx_folder = experiment_folder + "onnx_models/"
smtlib_folder = experiment_folder + "smtlib_benchmarks/"
logs_folder = experiment_folder + "logs/"
checkpoint_folder = experiment_folder + "training_checkpoints/"

if not os.path.exists(experiment_folder):
    os.mkdir(experiment_folder)

if not os.path.exists(onnx_folder):
    os.mkdir(onnx_folder)

if not os.path.exists(smtlib_folder):
    os.mkdir(smtlib_folder)

if not os.path.exists(logs_folder):
    os.mkdir(logs_folder)

if not os.path.exists(checkpoint_folder):
    os.mkdir(checkpoint_folder)


# ===== LOGGERS INSTANTIATION =====
#
#
#
logger_stream = logging.getLogger("pynever.strategies.training")
logger_file = logging.getLogger("benchmark_generation_file")

file_handler = logging.FileHandler(f"{logs_folder}benchmark_gen_logs.txt")
stream_handler = logging.StreamHandler()

file_handler.setLevel(logging.INFO)
stream_handler.setLevel(logging.INFO)

logger_file.addHandler(file_handler)
logger_stream.addHandler(stream_handler)

logger_file.setLevel(logging.INFO)
logger_stream.setLevel(logging.INFO)

#
#
#
# ===== FEATURE EXTRACTORS ARCHITECTURES DEFINITION =====
#
#
#
mnist_feature_extractor_arch = [

    (pyn_nodes.ConvNode, [8, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1, True]),  # Conv
    (pyn_nodes.ReLUNode, []),  # ReLU
    (pyn_nodes.MaxPoolNode, [(2, 2), (2, 2), (0, 0, 0, 0), (1, 1)]),  # MaxPool
    (pyn_nodes.ConvNode, [16, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1, True]),  # Conv
    (pyn_nodes.ReLUNode, []),  # ReLU
    (pyn_nodes.MaxPoolNode, [(2, 2), (2, 2), (0, 0, 0, 0), (1, 1)]),  # MaxPool
    (pyn_nodes.ConvNode, [32, (3, 3), (1, 1), (0, 0, 0, 0), (1, 1), 1, True]),  # Conv
    (pyn_nodes.ReLUNode, []),  # ReLU
    (pyn_nodes.MaxPoolNode, [(2, 2), (2, 2), (0, 0, 0, 0), (1, 1)]),  # MaxPool
    (pyn_nodes.FlattenNode, [])

]

#
#
#
# ===== PARAMETERS SELECTION =====
#
#
#
device = "mps"
mnist_benchmark_parameters = {

    # DATASET PARAMETERS
    "dataset_id": "mnist",
    "dataset_folder": "data/",
    "in_transform": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5)
    ]),

    # NETWORK PARAMETERS
    "feature_ext_arch": mnist_feature_extractor_arch,
    "classifier_archs": [[16], [32], [16, 8], [32, 16], [64]],
    "classifier_act_funs": [pyn_nodes.ReLUNode, pyn_nodes.SigmoidNode, pyn_nodes.TanhNode],
    "input_dimension": (1, 28, 28),
    "output_size": 10,
    "activation_on_output": False,

    # TRAINING PARAMETERS
    "validation_percentage": 0.3,
    "loss_fn": nn.CrossEntropyLoss(),
    "n_epochs": 10,
    "train_batch_size": 128,
    "validation_batch_size": 64,
    "opt_con": topt.Adam,
    "opt_params": {"lr": 0.001, "weight_decay": 0},

    # TESTING PARAMETERS
    "test_batch_size": 1,
    "save_results": False,
    "metric": pyn_training.PytorchMetrics.inaccuracy,
    "metric_params": {},

    # PROPERTY PARAMETERS
    "epsilons": [0.001, 0.01, 0.1]

}

#
#
#
# ===== EXPERIMENT INSTANCE SETUP =====
#
#
#
benchmarks_parameters = [mnist_benchmark_parameters]
smt_aux_vars = False
logger_file.info(f"benchmark_id,"
                 f"datetime,"
                 f"dataset_id,"
                 f"net_arch,"
                 f"activation_function,"
                 f"activation_on_output,"
                 f"validation_percentage,"
                 f"loss_fn,"
                 f"n_epochs,"
                 f"train_batch_size,"
                 f"validation_batch_size,"
                 f"optimizer,"
                 f"lr,"
                 f"weight_decay,"
                 f"test_percentage,"
                 f"test_batch_size,"
                 f"metric,"
                 f"test_metric_results,"
                 f"epsilon")

#
#
#
# ===== BENCHMARK GENERATION =====
#
#
#
benchmark_num = 0
for benchmark_params in benchmarks_parameters:

    # DATASET PARAMETERS
    dataset_id = benchmark_params["dataset_id"]
    dataset_folder = benchmark_params["dataset_folder"]
    transform = benchmark_params["in_transform"]

    # NETWORK PARAMETERS
    feature_ext_arch = benchmark_params["feature_ext_arch"]
    classifier_archs = benchmark_params["classifier_archs"]
    classifier_act_funs = benchmark_params["classifier_act_funs"]
    input_dimension = benchmark_params["input_dimension"]
    output_size = benchmark_params["output_size"]
    activation_on_output = benchmark_params["activation_on_output"]

    # TRAINING PARAMETERS
    validation_percentage = benchmark_params["validation_percentage"]
    loss_fn = benchmark_params["loss_fn"]
    n_epochs = benchmark_params["n_epochs"]
    train_batch_size = benchmark_params["train_batch_size"]
    validation_batch_size = benchmark_params["validation_batch_size"]
    checkpoint_root = checkpoint_folder
    opt_con = benchmark_params["opt_con"]
    opt_params = benchmark_params["opt_params"]

    # TESTING PARAMETERS
    test_batch_size = benchmark_params["test_batch_size"]
    save_results = benchmark_params["save_results"]
    metric = benchmark_params["metric"]
    metric_params = benchmark_params["metric_params"]

    # PROPERTY PARAMETERS
    epsilons = benchmark_params["epsilons"]

    logger_stream.info(f"BENCHMARKS {dataset_id}")

    if dataset_id == "mnist":
        training_set = pyn_data.TorchMNIST(dataset_folder, True, transform)
        test_set = pyn_data.TorchMNIST(dataset_folder, False, transform)
    else:
        raise NotImplementedError

    for act_fun in classifier_act_funs:

        input_sample = test_set.__getitem__(0)[0].unsqueeze(0).numpy()
        test_len = test_set.__len__()
        train_len = training_set.__len__()
        test_percentage = test_len / (test_len + train_len)
        logger_stream.info(f"Training Dataset Size: {train_len}")
        logger_stream.info(f"Test Dataset Size: {test_len}")
        logger_stream.info("")

        for cls_arch in classifier_archs:

            net_id = f"{dataset_id}_{act_fun.__name__}_{cls_arch}"
            network = pyn_net.SequentialNetwork(identifier=net_id, input_id="P")

            node_index = 0
            in_dim = input_dimension

            # Adding feature extractor layers to network
            for node_con, params in feature_ext_arch:

                new_node = node_con(f"{node_con.__name__}_{node_index}", in_dim, *params)
                network.add_node(new_node)
                in_dim = new_node.out_dim
                node_index += 1

            # Adding classifier layers to network
            for n_neurons in cls_arch:

                # FC Layer
                new_fc_node = pyn_nodes.FullyConnectedNode(identifier=f"FullyConnectedNode_{node_index}",
                                                           in_dim=in_dim, out_features=n_neurons)
                network.add_node(new_fc_node)
                node_index += 1

                # Activation Layer
                act_node = act_fun(identifier=f"{act_fun.__name__}_{node_index}", in_dim=new_fc_node.out_dim)
                network.add_node(act_node)
                in_dim = act_node.out_dim
                node_index += 1

            # Adding Output Layer to network
            fc_out_node = pyn_nodes.FullyConnectedNode(identifier=f"FullyConnectedNode_{node_index}", in_dim=in_dim,
                                                       out_features=output_size)
            network.add_node(fc_out_node)
            node_index += 1

            if activation_on_output:

                act_out_node = act_fun(identifier=f"{act_fun.__name__}_{node_index}", in_dim=fc_out_node.out_dim)
                network.add_node(act_out_node)

            train_strategy = pyn_training.PytorchTraining(optimizer_con=opt_con, opt_params=opt_params,
                                                          loss_function=loss_fn, n_epochs=n_epochs,
                                                          validation_percentage=validation_percentage,
                                                          train_batch_size=train_batch_size,
                                                          validation_batch_size=validation_batch_size,
                                                          checkpoints_root=checkpoint_root, precision_metric=metric,
                                                          device="mps")

            network = train_strategy.train(network=network, dataset=training_set)

            test_strategy = pyn_training.PytorchTesting(metric=metric, metric_params=metric_params,
                                                        test_batch_size=test_batch_size,
                                                        save_results=save_results)

            loss = test_strategy.test(network, test_set)
            logger_stream.info(f"Test Loss: {loss}")

            if save_results:
                outputs = np.array(test_strategy.outputs).squeeze()
                targets = np.array(test_strategy.targets).squeeze()
                losses = np.array(test_strategy.losses)
                logger_stream.info(f"{np.sum(losses)}, {len(targets)}")

            # Now we need to extract from the network the trained classifier and feature extractor
            fex_net = pyn_net.SequentialNetwork(f"{network.identifier}_fex", "P")
            cls_net = pyn_net.SequentialNetwork(f"{network.identifier}_cls", "P")

            current_node = network.get_first_node()
            is_flatten = False
            while current_node is not None:

                if not is_flatten:

                    fex_net.add_node(copy.deepcopy(current_node))
                    if isinstance(current_node, pyn_nodes.FlattenNode):
                        is_flatten = True

                else:
                    cls_net.add_node(copy.deepcopy(current_node))

                current_node = network.get_next_node(current_node)

            logger_stream.info(network.__str__())
            logger_stream.info(fex_net.__str__())
            logger_stream.info(cls_net.__str__())

            cls_input_dim = cls_net.get_first_node().in_dim[0]

            # SAVE ONNX MODELS
            onnx_net = pyn_conversion.ONNXConverter().from_neural_network(network).onnx_network
            onnx.save(onnx_net, onnx_folder + network.identifier + ".onnx")

            onnx_cls_net = pyn_conversion.ONNXConverter().from_neural_network(cls_net).onnx_network
            onnx.save(onnx_cls_net, onnx_folder + network.identifier + "_cls.onnx")

            # GENERATE SMTLIB PROPERTIES
            for epsilon in epsilons:

                benchmark_id = f"B_{benchmark_num:03d}"
                sanified_arch = str(cls_arch).replace(", ", "-")
                logger_file.info(
                    f"{benchmark_id},{benchmark_datetime},{dataset_id},{sanified_arch},{act_fun.__name__},{activation_on_output},"
                    f"{validation_percentage},"
                    f"{loss_fn.__class__.__name__},{n_epochs},{train_batch_size},{validation_batch_size},"
                    f"{opt_con.__name__},{opt_params['lr']},{opt_params['weight_decay']},"
                    f"{test_percentage},{test_batch_size},{metric.__name__},{loss},{epsilon}")

                smtlib_path_cvc = smtlib_folder + f"{benchmark_id}_cvc.smt2"
                smtlib_path_mathsat = smtlib_folder + f"{benchmark_id}_mathsat.smt2"
                net_property = utilities.generate_advrobustness_property(fex_net, cls_net, input_sample, epsilon)

                if smt_aux_vars:
                    utilities.to_smtlib(cls_net, net_property, smtlib_path_cvc, smt_solver="CVC5")
                    utilities.to_smtlib(cls_net, net_property, smtlib_path_mathsat, smt_solver="Mathsat")
                else:
                    utilities.to_smtlib_no_aux_var(cls_net, net_property, smtlib_path_cvc, smt_solver="CVC5")
                    utilities.to_smtlib_no_aux_var(cls_net, net_property, smtlib_path_mathsat, smt_solver="Mathsat")

                benchmark_num += 1


