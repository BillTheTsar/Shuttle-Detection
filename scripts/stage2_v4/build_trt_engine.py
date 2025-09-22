import tensorrt as trt
from pathlib import Path

TRT_LOGGER = trt.Logger(trt.Logger.INFO)

def build_engine(onnx_path: str, engine_path: str, fp16: bool = True):
    with trt.Builder(TRT_LOGGER) as builder, \
         builder.create_network(flags=1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, \
         trt.OnnxParser(network, TRT_LOGGER) as parser:

        # Config
        config = builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 32)  # 4 GB

        if fp16 and builder.platform_has_fast_fp16:
            print("✅ Using FP16 precision")
            config.set_flag(trt.BuilderFlag.FP16)

        # Load ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                print("❌ Failed to parse ONNX:")
                for i in range(parser.num_errors):
                    print(parser.get_error(i))
                return None

        # Dynamic batch profile
        profile = builder.create_optimization_profile()
        profile.set_shape("current_crop", (1, 3, 224, 224), (1, 3, 224, 224), (3, 3, 224, 224))
        profile.set_shape("past_crops",   (1, 3, 3, 224, 224), (1, 3, 3, 224, 224), (3, 3, 3, 224, 224))
        profile.set_shape("positions",    (1, 30, 3),          (1, 30, 3),          (3, 30, 3))
        config.add_optimization_profile(profile)

        # 🚨 New API: build serialized network
        serialized_engine = builder.build_serialized_network(network, config)
        if serialized_engine is None:
            print("❌ Failed to build serialized engine")
            return None

        # Save to disk
        with open(engine_path, "wb") as f:
            f.write(serialized_engine)
        print(f"✅ Saved TensorRT engine: {engine_path}")

        # Optional: return an actual engine object
        runtime = trt.Runtime(TRT_LOGGER)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        return engine

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
    onnx_path = str(PROJECT_ROOT / "onnx_exports/stage2_v4.onnx")
    engine_path = str(PROJECT_ROOT / "trt_engines/stage2_v4_fp16.engine")
    build_engine(onnx_path, engine_path, fp16=True)