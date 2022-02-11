import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import circuits_network as cn

layer_repetitions = 1000



class DepoNoiseModel(cirq.NoiseModel):
    """
    定义去极化噪声模型
    """
    def __init__(self, qubits, mean, stdev, seed=0):
        self.mean = mean
        self.stdev = stdev

        np.random.seed(seed)
        single_errors = {}
        for i in range(1):
            single_errors[i] = np.random.normal(self.mean[i], self.stdev[i])
        self.single_errors = single_errors

    def noisy_operation(self, op):

        return [op,
                cirq.ops.depolarize(self.single_errors[0]).on(op.qubits[0]),]


def build_gen_model(
        generator_qubits,
        gen_noise_qubits,
        dis_ancille_qubit,
        lr,
        prob,
        generator_initialization,
        gen_symbols,
        dis_param,
        backend=None):
    """
    生成模型定义，整个模型主要分成三部分。
    第一部分：
        经过加噪处理的量子线路，经过函数tfq.conver_to_tensor()转换为String格式输入到对应的带参数的生成量子线路，构成对应的layer，
        此带参数的生成量子线路是生成模型所需要训练的部分
    第二部分：
        辨别网络层构建，此时会把除了生成网络部分的量子线路构建好并转换为对应的String格式输入到此模型中，然后将第一部分生成网络层作为输入，
        输入到辨别网络中构建成一个完整的辨别网络层，以输入到最后的期望值获取层
    第三部分：
        对辨别网络辅助比特进行测量获取期望值，此layer的输入为第二部分的辨别网络层，通过SampledExpectation层实现对辅助比特的测量，
        并且获取一定重复次数的期望值，

    通过这样的方式构建生成模型而不是直接采用生成模型生成对应的量子态，将其所得的密度矩阵输入到辨别网络中去预测是true或fake的原因在于:
    cirq的量子构建过程，是无法将密度矩阵作为量子线路的输入，所以在构建的时候采用这样的方式，也因此对于生成模型的训练过程，输出值为辨别网络
    中辅助比特测量期望值，所以我们的训练目的是让通过生成网络的学习，使辨别网络能尽可能对其判断为True，所以生成模型在训练时，与一般的生成对抗
    网络不一致的地方在于，我们应该让这个网络的输出值越小越好而不是越大越好

    :param generator_qubits: 生成网络量子比特
    :param dis_ancille_qubit: 辨别网络辅助比特
    :param lr: 学习率
    :param generator_initialization: 生成网络参数初始化值
    :param gen_symbols: 生成网络参数符号
    :param gen_symbols: 生成网络参数符号
    :return:backend:计算辅助比特测量的期望值层所应用的量子机器的后端平台，如果定义为None时则采用默认的平台
    """

    gen_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    gen_layer = tfq.layers.AddCircuit()(gen_input, append=cn.get_noise_circuit(prob,generator_qubits)
                                                          + cn.get_gen_circuit(generator_qubits, gen_symbols))

    dis_circuit_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    dis_noise_layer = tfq.layers.AddCircuit()(dis_circuit_input, append=cn.get_noise_circuit(prob,gen_noise_qubits))

    dis_circuit_layer = tfq.layers.AddCircuit()(gen_layer, append=cn.get_noise_circuit(prob,gen_noise_qubits) +
                                                                  cn.dis_circuit(generator_qubits,gen_noise_qubits,dis_ancille_qubit,dis_param))

    expectation_output = tfq.layers.SampledExpectation(backend=backend)(
        dis_circuit_layer,
        symbol_names=gen_symbols,
        operators=cn.observe_operator(dis_ancille_qubit[0]),
        initializer=tf.constant_initializer(generator_initialization),
        repetitions=layer_repetitions)

    gen_model = tf.keras.Model(inputs=[gen_input, dis_circuit_input],
                               outputs=[expectation_output])

    gen_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    #gen_loss = lambda x, y: -tf.reduce_mean(y)
    gen_loss = tf.keras.losses.mean_squared_error
    gen_model.compile(optimizer=gen_optimizer, loss=gen_loss, loss_weights=[1])

    return gen_model


def build_dis_model(
        gen_qubits,
        true_qubits,
        gen_noise_qubits,
        gen_ancile_qubit,
        true_noise_qubits,
        true_ancile_qubit,
        discriminator_symbols,
        lr,
        prob,
        dis_initialization,
        backend=None):
    """
    辨别网络模型定义，对于整个网络的构建分成三个部分
    第一部分：
        输入到辨别网络的量子线路构建，根据输入的是生成量子比特或不加噪声的量子态分成两种情况分别构建对应的量子线路。量子条件生成对抗网络的辨别网络输入
        主要构建分成三部分：无噪声量子态+带噪声量子态(条件)+辅助比特或生成量子态（带噪声量子态）+带噪声量子态（条件）+辅助量子比特。而这三部分的量子态
        需要通过量子线路构建，通过tfq.conver_to_tensor()转换为String格式输入到模型
    第二部分：
        将第一部分的输入，分别构建输入为生成量子态的条件辨别网络层或无噪声量子态条件辨别网络层，后续需要将此作为输入输入到第三部分
    第三部分：
        将第二部分作为输入，此时根据两种情况分别输入SampledExpectation层，并且对辅助比特进行测量，最终获取定义好重复次数下的期望值

    整个辨别网络会一次性的输入True和Fake数据，输出为其对应的辅助比特测量期望值，此时采用测量算子对不同的情况，分别拟合不同的特征值，
    如：采用Z测量算子时，则True和Fake则分别对应特征值[1]和[-1]两种情况

    :param gen_qubits:生成量子比特
    :param true_qubits:无噪声量子比特
    :param gen_noise_qubits:生成情况下对应的条件带噪声量子比特，作为条件一起输入到辨别网络中
    :param gen_ancile_qubit:生成情况下对应的辅助量子比特
    :param true_noise_qubits:无噪声情况下对应的带噪声量子比特，作为条件一起输入到辨别网络中
    :param true_ancile_qubit:无噪声情况下对应的辅助量子比特
    :param discriminator_symbols:辨别网络参数符号
    :param lr:学习率
    :param dis_initialization:辨别网络参数初始化值
    :param backend:
    :return:计算辅助比特测量的期望值层所应用的量子机器的后端平台，如果定义为None时则采用默认的平台
    """
    gen_data_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)
    true_data_input = tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string)

    dis_gen_circuit_layer = tfq.layers.AddCircuit()(
        gen_data_input,
        append=(cn.get_noise_circuit(prob,gen_noise_qubits) +
                cn.dis_circuit(gen_qubits, gen_noise_qubits, gen_ancile_qubit, discriminator_symbols)))

    dis_true_circuit_layer = tfq.layers.AddCircuit()(
        true_data_input,
        append=(cn.get_noise_circuit(prob,true_noise_qubits)
                + cn.dis_circuit(true_qubits,true_noise_qubits,true_ancile_qubit,discriminator_symbols)))

    gen_expectation_output = tfq.layers.SampledExpectation(backend=backend)(
        dis_gen_circuit_layer,
        symbol_names=discriminator_symbols,
        operators=cn.observe_operator(gen_ancile_qubit[0]),
        initializer=tf.constant_initializer(dis_initialization),
        repetitions=layer_repetitions)

    true_expectation_output = tfq.layers.SampledExpectation(backend=backend)(
        dis_true_circuit_layer,
        symbol_names=discriminator_symbols,
        operators=cn.observe_operator(true_ancile_qubit[0]),
        initializer=tf.constant_initializer(dis_initialization),
        repetitions=layer_repetitions)

    dis_model = tf.keras.Model(
        inputs=[gen_data_input,true_data_input],
        outputs=[gen_expectation_output,true_expectation_output])

    dis_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
    dis_loss = lambda x, y: tf.reduce_mean(y)
   # dis_loss = tf.keras.losses.mean_squared_error
    dis_model.compile(optimizer=dis_optimizer, loss=dis_loss, loss_weights=[1,1])

    return dis_model

