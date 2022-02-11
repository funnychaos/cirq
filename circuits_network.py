import cirq
import sympy
import numpy as np

def get_circuit_param_symbol(gen_qubits,dis_qubits):
    """
    定义生成网络与辨别网络所需的参数符号，根据已经定义好的量子门结构直接输入对应的符号数
    所以
    :param gen_qubits: 生成网络比特数
    :param dis_qubits: 辨别网络比特数
    :return: 生成网络与辨别网络符号列表
    """
    gen_param_symbols = []
    dis_param_symbols = []

    for i in range(2*gen_qubits):
        gen_param_symbols.append(sympy.Symbol("G_param{!r}".format(i)))
    for i in range(2*dis_qubits):
        dis_param_symbols.append(sympy.Symbol("D_param{!r}".format(i)))

    return gen_param_symbols, dis_param_symbols


def get_ghz_circuit(qubits):
    """
    获取标准GHZ量子线路
    :param qubits:
    :return: 返回GHZ量子线路
    """
    ghz_circuit = cirq.Circuit(
        cirq.H(qubits[0]),
        cirq.CNOT(qubits[0], qubits[1]),
        cirq.CNOT(qubits[1], qubits[2]),
        cirq.CNOT(qubits[2], qubits[3]))
    return ghz_circuit


def ancille_init_circuit(ancile_qubit):
    """
    定义辨别网络辅助比特初始化量子线路
    :param ancile_qubit: 辅助比特
    :return: 初始化后的量子比特
    """
    ancille_init_circuit = cirq.Circuit(cirq.H(ancile_qubit[0]))
    return ancille_init_circuit

def get_layer_circuit(gen_qubits,param,index):
    param_index = 2 * index
    gen_layer_circuit = cirq.Circuit(
        cirq.rx(param[param_index+0]).on(gen_qubits[0]),
        cirq.rx(param[param_index+1]).on(gen_qubits[1]),
        cirq.rx(param[param_index+2]).on(gen_qubits[2]),
        cirq.rx(param[param_index+3]).on(gen_qubits[3]),
        cirq.CNOT(gen_qubits[0], gen_qubits[1]),
        cirq.CNOT(gen_qubits[1], gen_qubits[2]),
        cirq.CNOT(gen_qubits[2], gen_qubits[3]),
        cirq.CNOT(gen_qubits[3], gen_qubits[0]),
        cirq.rz(param[param_index+4]).on(gen_qubits[0]),
        cirq.rz(param[param_index+5]).on(gen_qubits[1]),
        cirq.rz(param[param_index+6]).on(gen_qubits[2]),
        cirq.rz(param[param_index+7]).on(gen_qubits[3]))

    return gen_layer_circuit


def get_gen_circuit(gen_qubits, param, layer):
    """
    定义生成网络量子线路，采用RX+CNOT+RZ量子线路结构
    :param gen_qubits: 生成网络量子比特
    :param param: 传入生成网络的参数或参数符号
    :return: 生成网络量子线路
    """
    gen_circuit = cirq.Circuit()
    for i in range(layer):
        layer_circuit = get_layer_circuit(gen_qubits,param,4*i)
        gen_circuit += layer_circuit

    return gen_circuit


def dis_circuit(true_or_gen_qubit, noise_qubit, ancile_qubit, param_symbols):
    """
    定义辨别网络量子线路，采用RX+CNOT+RZ量子线路结构
    :param true_or_gen_qubit: 不带噪声的量子比特或经过生成网络的量子比特
    :param noise_qubit:作为对应条件，带噪声的量子比特
    :param ancile_qubit: 辅助比特，用于辨别网络测量使用
    :param param_symbols: 传入辨别网络的参数或参数符号
    :return: 辨别网络
    """
    dis_circuit = cirq.Circuit(
        # rx门
        cirq.rx(param_symbols[0]).on(true_or_gen_qubit[0]),
        cirq.rx(param_symbols[1]).on(true_or_gen_qubit[1]),
        cirq.rx(param_symbols[2]).on(true_or_gen_qubit[2]),
        cirq.rx(param_symbols[3]).on(true_or_gen_qubit[3]),
        cirq.rx(param_symbols[4]).on(noise_qubit[0]),
        cirq.rx(param_symbols[5]).on(noise_qubit[1]),
        cirq.rx(param_symbols[6]).on(noise_qubit[2]),
        cirq.rx(param_symbols[7]).on(noise_qubit[3]),
        cirq.rx(param_symbols[8]).on(ancile_qubit[0]),
        # CNOT门
        cirq.CNOT(true_or_gen_qubit[0], true_or_gen_qubit[1]),
        cirq.CNOT(true_or_gen_qubit[1], true_or_gen_qubit[2]),
        cirq.CNOT(true_or_gen_qubit[2], true_or_gen_qubit[3]),
        cirq.CNOT(true_or_gen_qubit[3], noise_qubit[0]),
        cirq.CNOT(noise_qubit[0], noise_qubit[1]),
        cirq.CNOT(noise_qubit[1], noise_qubit[2]),
        cirq.CNOT(noise_qubit[2], noise_qubit[3]),
        cirq.CNOT(noise_qubit[3], ancile_qubit[0]),
        # rz门
        cirq.rz(param_symbols[9]).on(true_or_gen_qubit[0]),
        cirq.rz(param_symbols[10]).on(true_or_gen_qubit[1]),
        cirq.rz(param_symbols[11]).on(true_or_gen_qubit[2]),
        cirq.rz(param_symbols[12]).on(true_or_gen_qubit[3]),
        cirq.rz(param_symbols[13]).on(noise_qubit[0]),
        cirq.rz(param_symbols[14]).on(noise_qubit[1]),
        cirq.rz(param_symbols[15]).on(noise_qubit[2]),
        cirq.rz(param_symbols[16]).on(noise_qubit[3]),
        cirq.rz(param_symbols[17]).on(ancile_qubit[0]))
    return dis_circuit


def get_noise_circuit(prob,qubits):
    """
    噪声处理量子线路，由于使用cirq原生的加噪声函数如：cirq.depolarize(p)时，在后面如果使用tfq.convert_to_tensor(circuit)转换格式输入
    到model时，无法实现序列化，转换为对应的string格式输入到model，所以根据去极化噪声对应原理定义此加噪声量子线路。
    函数一开始获取（0,1）之间的平均分布的随机数，将其与输入的prob对比，如果小于此值，表示需要在原有的量子线路基础上增加噪声部分，否则不做任何操作
    :param prob: 定义噪声量子线路加入噪声的概率
    :param circuit: 输入已经初始化后的量子线路
    :return: 噪声线路
    """
    # p = np.random.uniform(0, 1)
    # if p >= prob:
    #     noise_circuit = circuit
    # else:
    #     noise_circuit = circuit + cirq.Circuit(
    #         cirq.X.on_each(*circuit.all_qubits()),
    #         cirq.Y.on_each(*circuit.all_qubits()),
    #         cirq.Z.on_each(*circuit.all_qubits())
    #     )
    noise_circuit = cirq.Circuit(
        cirq.depolarize(prob).on_each(qubits)
    )

    return noise_circuit

def observe_operator(qubit):
    """
    测量算子
    :param qubit: 需要测量的比特
    :return: 测量算子
    """
    observer = [cirq.Z(qubit)]

    return observer
