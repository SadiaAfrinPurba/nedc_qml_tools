#!/usr/bin/env python

# file: $(NEDC_NFC)/class/python/nedc_qml_tools/test/test_nedc_qml_tools.py
#
# revision history: 
#
# 20250203 (SP): initial version
#
# This file contains the test cases for the nedc_qml_tools module.
#  
#------------------------------------------------------------------------------

# import reqired system modules
#
import unittest
from unittest import TestCase
import os
import numpy as np
import warnings

# import required NEDC modules
#
import nedc_debug_tools as ndt
import nedc_qml_tools as nqt
import nedc_qml_tools_constants as const

dbgl = ndt.Dbgl()

#------------------------------------------------------------------------------
#
# global variables are listed here
#
#------------------------------------------------------------------------------

# set the filename using basename
#
__FILE__ = os.path.basename(__file__)

# define variables to handle option names and values. For each of these,
# we list the parameter name, the allowed values, and the default values.
#

#------------------------------------------------------------------------------
#
# classes listed here
#
#------------------------------------------------------------------------------

class TestNedcQmlTools(TestCase):
    def setUp(self):
        # ignore warnings
        #
        warnings.filterwarnings('ignore')

    def test_qsvm_when_fm_zz_kernel_fidelity(self):
    
        # arrange
        #        
        self.samples = np.array([
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

        self.labels = np.array(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
        
        expected_predictions = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
        expected_err_rate = 9.99
        
        # act
        #
        qml = nqt.QML(model_name=const.MODEL_NAME_QSVM,
                      kernel_name=const.KERNEL_NAME_FIDELITY,
                      encoder_name=const.ENCODER_NAME_ZZ,
                      provider_name=const.PROVIDER_NAME_QISKIT)
        qml.fit(self.samples, self.labels)
        actual_err_rate, actual_predictions = qml.score(self.samples, self.labels)

        # assert
        #
        self.assertEqual(actual_predictions.shape[0], 10)
        # self.assertListEqual(actual_predictions.tolist(), expected_predictions.tolist())
        self.assertAlmostEqual(actual_err_rate, expected_err_rate, places=1)
        
    def test_qsvm_when_fm_z_kernel_fidelity(self):
    
        # arrange
        #        
        self.samples = np.array([
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

        self.labels = np.array(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
        
        expected_predictions = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
        expected_err_rate = 9.99
        
        # act
        #
        qml = nqt.QML(model_name=const.MODEL_NAME_QSVM,
                      kernel_name=const.KERNEL_NAME_FIDELITY,
                      encoder_name=const.ENCODER_NAME_Z,
                      provider_name=const.PROVIDER_NAME_QISKIT)
        qml.fit(self.samples, self.labels)
        actual_err_rate, actual_predictions = qml.score(self.samples, self.labels)

        # assert
        #
        self.assertEqual(actual_predictions.shape[0], 10)
        # self.assertListEqual(actual_predictions.tolist(), expected_predictions.tolist())
        self.assertAlmostEqual(actual_err_rate, expected_err_rate, places=1)
    
    def test_qnn_when_fm_zz_ansatz_real_amplitudes(self):
        
        # arrange
        #    
        self.samples = np.array([
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

        self.labels = np.array(np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]))
        
        expected_predictions = np.array([0, 0, 0, 1, 0, 1, 1, 1, 1, 1])
        expected_err_rate = 9.99
        
        # act
        #
        qml = nqt.QML(model_name=const.MODEL_NAME_QNN,
                      ansatz=const.ANSATZ_NAME_REAL_AMPLITUDES,
                      encoder_name=const.ENCODER_NAME_ZZ,
                      provider_name=const.PROVIDER_NAME_QISKIT,
                      optim_name=const.OPTIM_NAME_COBYLA,
                      n_qubits=2)
        qml.fit(self.samples, self.labels)
        actual_err_rate, actual_predictions = qml.score(self.samples, self.labels)
        
        # assert
        #
        self.assertEqual(actual_predictions.shape[0], 10)
        # self.assertListEqual(sorted(actual_predictions.tolist()), sorted(expected_predictions.tolist()))
        self.assertAlmostEqual(actual_err_rate, expected_err_rate, places=1)
        
    def test_seed_consistency(self):
        dbgl1 = ndt.Dbgl()
        result1 = np.random.normal(0, 1, 5)
        print(result1)
        
        dbgl2 = ndt.Dbgl()
        result2 = np.random.normal(0, 1, 5)
        print(result1)

        self.assertListEqual(result1.tolist(), result2.tolist())
        
if __name__ == '__main__':
    unittest.main()
    
#
# end of file
#------------------------------------------------------------------------------



