# NEDC QML Tools - Quantum Machine Learning Framework

## Architecture Overview

                            QML (High-Level Interface)
                                      |
                                      |
                    +-----------------+-----------------+
                    |                 |                 |
              QuantumModel    QuantumProvider    HardwareSpec
                    |                 |                 |
            +-------+-------+  +------+------+    Hardware Config
            |       |       |  |             |
          QSVM    QNN    ...  Qiskit      Future
                              Provider    Providers
-------------------------------------------------------------------------------

## Quick Start Example

from nedc_qml_tools import QML
import numpy as np

# create training data
#
X_train = np.array([[0,0], [0,1], [1,0], [1,1]])
y_train = np.array([0, 1, 1, 0])

# initialize QML with QSVM and Qiskit
#
qml = QML(
    model_name='qsvm',
    provider_name='qiskit',
    encoder_name='zz',
    kernel_name='fidelity',
    n_qubits=2,
    featuremap_reps=2
)

# train and predict
#
trained_model = qml.fit(X_train, y_train)

predictions = qml.predict(X_train)
>> [0, 1, 1, 0]
error_rate, prections = qml.score(X_train, X_train)
>> 0.00, [0, 1, 1, 0]

-------------------------------------------------------------------------------

## Extending the Framework

### Adding a New Encoder

Lock the files:

- isip_co nedc_qml_tools_constants.py
- isip_co nedc_qml_base_providers_tools.py
- isip_co nedc_qml_providers_tools.py

1) Add constant in nedc_qml_tools_constants.py
   ENCODER_NAME_NEW = 'new'

2a) Add abstract method in nedc_qml_base_providers_tools.py

   @abstractmethod
   def get_new_encoder(self):
       raise NotImplementedError

2b) Update ENCODER_COMPUTERS dictionary:
    
    ENCODER_COMPUTERS = {
    QISKIT: {
        ZZ: "get_zz_encoder",
        Z: "get_z_encoder",
        NEW: "get_new_encoder"
        }
    }
    Follow similar pattern for adding a new kernel, ansatz or optimizer:

    Kernel functions (KERNEL_COMPUTERS)
    Ansatz circuits (ANSATZ_COMPUTERS)
    Optimizers (OPTIM_COMPUTERS)
    Hardware backends (HARDWARE_SPECS)

3) Implement concrete method in nedc_qml_providers_tools.py

    Assuming we are adding a Qiskit encoder/featuremap, so adding the
    concrete implementation in QiskitProvider class.

    def get_new_encoder(self):
        """
        method: get_new_encoder
            
        arguments:
        none
            
        return: 
        ZZFeatureMap: Qiskit New circuit
            
        description:
        creates a New feature map encoder with specified parameters
        """
        
        if dbgl == ndt.FULL:
            print("Creating new encoder")
        
        return NewEncoder()

4) Run `make install` 

Checked in the changes by running:

- isip_ci nedc_qml_tools_constants.py -> add a log message, and press .
- isip_ci nedc_qml_base_providers_tools.py -> add a log message, and press .
- isip_ci nedc_qml_providers_tools.py -> add a log message, and press .

### [Optional]

cd tests

1) Add a new test for the newly added module in test_nedc_qml_tools.py
2) run_tests.sh

-------------------------------------------------------------------------------

### Adding a New Qunatum model
Lock the files:

- isip_co nedc_qml_tools_constants.py
- isip_co nedc_qml_tools.py

1) Add constant in nedc_qml_tools_constants.py
   MODEL_NAME_NEW = 'new'

2a) Add the new quantum model in nedc_qml_tools.py
    
    class NEW(nbpt.QuantumModel):
        """
        Description of the NEW model
        """
        def __init__(self):
            """
            method: constructor
            arguments: none
            return: none
            description: initializes New
            """
            if dbgl == ndt.FULL:
                print("%s (line: %s) %s: initializing NEW" % 
                    (__FILE__, ndt.__LINE__, ndt.__NAME__))
            
            # call the constructor of the parent class which is QuantumModel
            #
            super().__init__()
        
        # end of constructor
        #
        
        def __repr__(self) -> str:
            """
            method: __repr__
            arguments: none
            return: str
            description: returns string representation of NEW instance
            """
            return f"{self.__class__.__name__}(provider={self.provider.__class__.__name__ if self.provider else None})"
        
        def fit(self, X, y=None):
            """
            method: fit
            arguments:
            X: training data features
            y: training data labels
            return: none
            description: trains NEW model
            """
            if dbgl == ndt.FULL:
                print("%s (line: %s) %s: computing kernel and fitting NEW" % 
                    (__FILE__, ndt.__LINE__, ndt.__NAME__))
        
            
            # exit gracefully
            #
            
        def predict(self, X, y=None):
            """
            method: predict
            arguments:
            X: test data features
            y: test data labels
            return: predictions
            description: predicts labels using NEW
            """
            if dbgl == ndt.FULL:
                print("%s (line: %s) %s: predicting labels using NEW" % 
                    (__FILE__, ndt.__LINE__, ndt.__NAME__))
            
            
            # exit gracefully: return the predicted labels
            #
            return predictions     
    #
    # end of NEW

2b) Register the new model at the bottom of the file
    Registry.register_model(const.MODEL_NAME_NEW, NEW)

3) Run `make install` 

Checked in the changes by running:

- isip_ci nedc_qml_tools_constants.py -> add a log message, and press .
- isip_ci nedc_qml_tools.py -> add a log message, and press .

-------------------------------------------------------------------------------

### Adding a New Qunatum Provider
Lock the files:

- isip_co nedc_qml_tools_constants.py
- isip_co nedc_qml_providers_tools.py
- isip_co nedc_qml_base_providers_tools.py
- isip_co nedc_qml_tools.py

1) Add constant in nedc_qml_tools_constants.py
   PROVIDER_NAME_NEW = 'new'

2) Add the new quantum provider class and necessary methods in   nedc_qml_providers_tools.py
    
    class NewProvider(QuantumProvider):
    """
    Description
    """
    
    def __init__(self, params: QMLParams):
        """
        method: __init__
        arguments: params (QMLParams): Parameters for quantum machine learning
        operations.
        return: none
        description: Initializes the QiskitProvider object with the given
        parameters.
        """
        
        # call the parent class constructor with the given parameters
        #
        super().__init__(params=params)
        self.params = params
        
    # end of constructor
    #
        
    def __repr__(self) -> str:
        """
        method: __repr__
        arguments: none
        return: str
        description: returns string representation of NewProvider instance
        """
        return f"NewProvider(params={self.params})"
    
    #--------------------------------------------------------------------------
    #
    # concrete methods listed here : 
    #
    #--------------------------------------------------------------------------
    def get_basic_simulator(self):     
        # exit gracefully: 
        #
        return a
    
    def method2(self): 
        # exit gracefully: 
        #
        return b
    
3) Update the HARDWARE_SPECS dictionary with proper simulator function name in
nedc_qml_base_providers_tools.py. For example: 
    
    HARDWARE_SPECS = {,
                  NewProvider: {const.HARDWARE_NAME_CPU: "get_basic_simulator"}
    }

4) Register the new quantum provider at the bottom of nedc_qml_tools.py file
    Registry.register_provider(const.PROVIDER_NAME_NEW, nqpt.NewProvider)

5) Run `make install` 

Checked in the changes by running:

- isip_ci nedc_qml_tools_constants.py -> add a log message, and press .
- isip_ci nedc_qml_providers_tools.py -> add a log message, and press .
- isip_ci nedc_qml_base_providers_tools.py -> add a log message, and press .
- isip_ci nedc_qml_tools.py -> add a log message, and press .


