import cirq
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
import matplotlib.pyplot as plt
import circuits_network as cn
import model

noise_prob = 0.4
gen_qubits_len = 4
dis_qubits_len = 9
learning_rate = 0.01
train_samples = 100
test_samples = 50
dis_model_epochs = 1000
gen_model_epochs = 1000
layers = 2

# test-data
# train_samples = 5
# test_samples = 50
# dis_model_epochs = 100
# gen_model_epochs = 100




def load_noise_data(gen_qubits, gen_noise_qubits, true_noise_qubits):
    """
    获取带噪声量子线路
    :param gen_qubits: 输入到生成网络之前的量子态需要进行加噪操作
    :param gen_noise_qubits: 输入到辨别网络为生成量子态时，所对应的条件带噪量子态
    :param true_noise_qubits: 输入到辨别网络为无噪声量子态时，所对应的条件带噪量子态
    :return: 生成所需要的带噪量子态的量子线路
    """
    prob = np.random.uniform(0, noise_prob)
    ghz_circuit = cn.get_ghz_circuit(gen_qubits)
    ghz_gen_noise_circuit = cn.get_ghz_circuit(gen_noise_qubits)
    ghz_pure_noise_circuit = cn.get_ghz_circuit(true_noise_qubits)

    noise_circuit = cn.get_noise_circuit(prob, ghz_circuit)
    gen_noise_circuit = cn.get_noise_circuit(prob, ghz_gen_noise_circuit)
    pure_noise_circuit = cn.get_noise_circuit(prob, ghz_pure_noise_circuit)

    return noise_circuit, gen_noise_circuit, pure_noise_circuit


def load_true_data(true_qubits):
    """
    获得无噪声量子线路，量子态进过GHZ量子线路初始化后不经过然和加噪操作
    :param true_qubits:
    :return:
    """
    ghz_circuit = cn.get_ghz_circuit(true_qubits)

    return ghz_circuit

def quantum_data_fidelity(gen_circuit,pure_circuit):
    """
    计算两个量子比特保真度
    :param gen_circuit:
    :param pure_circuit:
    :return:
    """
    sim_matrix = cirq.DensityMatrixSimulator()
    sim = cirq.Simulator()

    gen_matrix = sim_matrix.simulate(gen_circuit).final_density_matrix
    pure_matrix = sim.compute_amplitudes(pure_circuit,[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
    pure_matrix = np.array(pure_matrix)
    pure_matrix_t = pure_matrix.reshape(16, 1)
    result = pure_matrix.dot(gen_matrix).dot(pure_matrix_t) ** 0.5

    return result

def data_save(file_name,data,key):
    """
    将训练过程中的数据保存到本地文件
    :param file_name:
    :param data:
    :param key:
    :return:
    """
    stream = open(file_name, 'a')
    data_stream = {key,data}
    stream.write(str(data_stream))
    stream.write("\n")
    stream.close()
    return


def train_cGAN():
    """
    条件量子生成对抗网络训练
    :return: 返回训练好的网络 gen_model和dis_model
    """
    dis_loss_history = []
    gen_loss_history = []

    dis_param_history = []
    gen_param_history = []

    #量子比特定义
    gen_qubits = cirq.GridQubit.rect(1, gen_qubits_len)
    gen_noise_condition_qubits = cirq.GridQubit.rect(1, gen_qubits_len)
    gen_ancille_qubits = cirq.GridQubit.rect(1, 1)
    pure_qubits = cirq.GridQubit.rect(1, gen_qubits_len)
    pure_noise_condition_qubits = cirq.GridQubit.rect(1, gen_qubits_len)
    pure_ancille_qubits = cirq.GridQubit.rect(1, 1)

    label_true = np.array([[1]], dtype=np.float32)
    label_fake = np.array([[-1]], dtype=np.float32)

    gen_param_symbols, dis_param_symbols = cn.get_circuit_param_symbol(gen_qubits_len,dis_qubits_len)

    #构建模型
    generator_initialization = np.random.uniform(0, 2*math.pi, 2 * gen_qubits_len * layers)
    discriminator_initialization = np.random.uniform(0, 1*math.pi, 2 * dis_qubits_len)
    gen_model = model.build_gen_model(gen_qubits,
                                      gen_ancille_qubits,
                                      learning_rate,
                                      generator_initialization,
                                      gen_param_symbols,
                                      None)
    dis_model = model.build_dis_model(gen_qubits,
                                pure_qubits,
                                gen_noise_condition_qubits,
                                gen_ancille_qubits,
                                pure_noise_condition_qubits,
                                pure_ancille_qubits,
                                dis_param_symbols,
                                learning_rate,
                                discriminator_initialization,
                                None)

    for i in range(train_samples):
        gen_param = gen_model.trainable_variables[0].numpy()
        dis_param = dis_model.trainable_variables[0].numpy()


        train_noise_circuit, train_noise_gen_circuit, train_noise_pure_circuit = load_noise_data(gen_qubits,
                                                                                                 gen_noise_condition_qubits,
                                                                                                 pure_noise_condition_qubits)
        train_pure_circuit = load_true_data(pure_qubits)

        gen_model_gen_circuit_input_tensor = tfq.convert_to_tensor([train_noise_circuit])

        gen_model_dis_circuit_input_tensor = tfq.convert_to_tensor([train_noise_gen_circuit +
                                                                    cn.ancille_init_circuit(gen_ancille_qubits) +
                                                                    cn.dis_circuit(gen_qubits, gen_noise_condition_qubits,
                                                                                gen_ancille_qubits, dis_param)])
        gen_model_return = gen_model([gen_model_gen_circuit_input_tensor, gen_model_dis_circuit_input_tensor])

        dis_gen_input = tfq.convert_to_tensor([cn.get_gen_circuit(gen_qubits, gen_param) +
                                               train_noise_gen_circuit +
                                               cn.ancille_init_circuit(gen_ancille_qubits) +
                                               cn.dis_circuit(gen_qubits, gen_noise_condition_qubits, gen_ancille_qubits,
                                                           dis_param)])

        dis_true_input = tfq.convert_to_tensor([train_pure_circuit +
                                                train_noise_pure_circuit +
                                                cn.ancille_init_circuit(pure_ancille_qubits) +
                                                cn.dis_circuit(pure_qubits, pure_noise_condition_qubits,
                                                            pure_ancille_qubits, dis_param)])
        dis_return = dis_model([dis_gen_input,dis_true_input])

        dis_param_history.append(dis_param)
        gen_param_history.append(gen_param)


        dis_history = dis_model.fit(x=[dis_gen_input,dis_true_input],
                                    y=[label_fake,label_true],
                                    epochs=dis_model_epochs)

        dis_model.trainable = False
        gen_history = gen_model.fit(x=[gen_model_gen_circuit_input_tensor, gen_model_dis_circuit_input_tensor],
                                    y=[label_true],
                                    epochs=gen_model_epochs)
        dis_model.trainable = True

        dis_loss_history.append(dis_history.history['loss'])
        gen_loss_history.append(gen_history.history['loss'])


    #保存参数和历史数据
    index = train_samples * dis_model_epochs
    dis_loss_history = (np.array(dis_loss_history)).reshape((index,1))
    gen_loss_history = (np.array(gen_loss_history)).reshape((index,1))

    time_now = str(time.strftime("%Y-%m-%d %H-%M-%S", time.localtime()))
    file_name = 'data_saver/model data ' + time_now + '.txt'
    data_save(file_name,str(dis_param_history),"dis_param_history")
    data_save(file_name, str(gen_param_history), "gen_param_history")
    data_save(file_name, str(dis_loss_history), "dis_loss_history")
    data_save(file_name, str(gen_loss_history), "gen_loss_history")


    plt.plot(dis_loss_history)
    plt.plot(gen_loss_history)
    plt.title("Learning to Control a Qubit")
    plt.xlabel("Iterations")
    plt.ylabel("Error in Control")
    plt.show()

    return gen_model,dis_model

train_cGAN()

def predict_cGAN():
    gen_model = train_cGAN()
    gen_param = gen_model.trainable_variables[0].numpy()

    noise_qubits = cirq.GridQubit.rect(1,4)
    pure_qubits = cirq.GridQubit.rect(2,4)
    ghz_noise_circuit = cn.get_ghz_circuit(noise_qubits)
    ghz_pure_circuit = cn.get_ghz_circuit(pure_qubits)

    fidelity_list = []

    for i in range(test_samples):
        prob = np.random.uniform(0, noise_prob)
        noise_circuit = cn.get_noise_circuit(prob,ghz_noise_circuit)
        gen_circuit = cn.get_gen_circuit(noise_qubits,gen_param) + noise_circuit
        fidelity = quantum_data_fidelity(gen_circuit,ghz_pure_circuit)
        fidelity_list.append(fidelity)

    # plt.plot(fidelity_list)
    # plt.title("Learning to Control a Qubit")
    # plt.xlabel("Iterations-prediction")
    # plt.ylabel("predict")
    # plt.show()

    return


