import numpy as np
from nedc_qml_tools import QML



samples =  np.array([
                    [-0.54585317,  0.45872878],
                    [ 0.19559004, -0.41765453],
                    [ 0.37622609, -1.21337474],
                    [ 0.41991704, -2.62169056],
                    [ 0.72306177, -0.89542115],
                    [ 4.82940720,  6.71913411],
                    [ 7.57186778,  4.80661997],
                    [ 5.08151120,  5.45632660],
                    [ 4.15677117,  6.52675315],
                    [ 5.44251527,  5.21502550]
                    ])

labels = np.array(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))

#------------------------------------------------------------------------------
# TEST QSVM
# #------------------------------------------------------------------------------

qsvm = QML(model_name='qsvm', 
           provider_name='qiskit', 
           encoder_name='zz',   
           kernel_name='fidelity', 
           n_qubits=2, 
           featuremap_reps=2, 
           n_classes=2)


qsvm.fit(samples, labels)
err_rate, predictions = qsvm.score(samples, labels)
print(f"[QSVM] Predictions: {predictions}")
print(f"[QSVM] Error Rate: {err_rate:.2f}%")

#------------------------------------------------------------------------------
# TEST QNN
# #------------------------------------------------------------------------------

qsvm = QML(model_name='qnn', 
           provider_name='qiskit', 
           encoder_name='zz',   
           ansatz='real_amplitudes', 
           n_qubits=2, 
           featuremap_reps=2,
           ansatz_reps=2, 
           n_classes=2,
           optim_name='cobyla',
           optim_max_steps=50)


qsvm.fit(samples, labels)
err_rate, predictions = qsvm.score(samples, labels)
print(f"[QNN] Predictions: {predictions}")
print(f"[QNN] Error Rate: {err_rate:.2f}%")

#------------------------------------------------------------------------------
# TEST QRBM
# #------------------------------------------------------------------------------

qrbm = QML(provider_name='dwave', 
           model_name='qrbm',
           encoder_name='bqm',
           shots=2, 
           n_hidden=2, 
           n_visible=2, 
           epochs=2, 
           lr=0.1, 
           n_neighbors=2)


qrbm.fit(samples, labels)
err_rate, predictions = qrbm.score(samples, labels)
print(f"[QNN] Predictions: {predictions}")
print(f"[QNN] Error Rate: {err_rate:.2f}%")


