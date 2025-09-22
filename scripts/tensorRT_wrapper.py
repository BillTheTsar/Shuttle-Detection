import tensorrt as trt
import torch
import numpy as np
import cuda.cudart as cudart

TRT_LOGGER = trt.Logger(trt.Logger.INFO)


class TRTWrapperTorch:
    def __init__(self, engine_path, input_names, output_names, device="cuda:0"):
        self.device = torch.device(device)

        # Load engine
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(TRT_LOGGER)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        self.input_names = input_names
        self.output_names = output_names

        # Map tensor names → indices manually
        self.name_to_index = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.name_to_index[name] = i

        # Keep metadata
        self.bindings = {}
        for name in self.input_names + self.output_names:
            if name not in self.name_to_index:
                raise ValueError(f"Tensor {name} not found in engine bindings!")
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = tuple(self.engine.get_tensor_shape(name))
            self.bindings[name] = {
                "index": self.name_to_index[name],
                "dtype": dtype,
                "shape": shape,
                "device_mem": None
            }

    def __call__(self, inputs: dict):
        # 1. Prepare inputs
        for name, tensor in inputs.items():
            assert name in self.input_names
            arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)

            # Dynamic shape support
            self.context.set_input_shape(name, arr.shape)

            nbytes = arr.nbytes
            if (self.bindings[name]["device_mem"] is None or
                self.bindings[name]["shape"] != arr.shape):
                err, dptr = cudart.cudaMalloc(nbytes)
                if err != 0:
                    raise RuntimeError(f"cudaMalloc failed for {name}")
                self.bindings[name]["device_mem"] = dptr
                self.bindings[name]["shape"] = arr.shape

            cudart.cudaMemcpy(self.bindings[name]["device_mem"], arr.ctypes.data, nbytes,
                              cudart.cudaMemcpyKind.cudaMemcpyHostToDevice)

        # 2. Allocate outputs
        for name in self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self.bindings[name]["dtype"]
            nbytes = np.prod(shape) * np.dtype(dtype).itemsize

            if (self.bindings[name]["device_mem"] is None or
                self.bindings[name]["shape"] != shape):
                err, dptr = cudart.cudaMalloc(nbytes)
                if err != 0:
                    raise RuntimeError(f"cudaMalloc failed for output {name}")
                self.bindings[name]["device_mem"] = dptr
                self.bindings[name]["shape"] = shape

        # 3. Build bindings list
        bindings_list = [0] * self.engine.num_io_tensors
        for name, info in self.bindings.items():
            bindings_list[info["index"]] = int(info["device_mem"])

        # 4. Execute
        self.context.execute_v2(bindings_list)

        # 5. Copy outputs back
        outputs = {}
        for name in self.output_names:
            shape = self.bindings[name]["shape"]
            dtype = self.bindings[name]["dtype"]
            nbytes = np.prod(shape) * np.dtype(dtype).itemsize
            host_arr = np.empty(shape, dtype=dtype)
            cudart.cudaMemcpy(host_arr.ctypes.data, self.bindings[name]["device_mem"],
                              nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost)
            outputs[name] = torch.from_numpy(host_arr).to(self.device)

        return outputs