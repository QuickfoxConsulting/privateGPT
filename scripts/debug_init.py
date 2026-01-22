from llama_index.core.tools import BaseTool, ToolMetadata
import inspect

print("BaseTool MRO:", BaseTool.mro())
print("BaseTool init signature:", inspect.signature(BaseTool.__init__))

try:
    print("Attempting instantiation with metadata kwarg...")
    meta = ToolMetadata(name="test", description="desc")
    t = BaseTool(metadata=meta)
    print("Success kwarg")
except Exception as e:
    print(f"Failed kwarg: {e}")

try:
    print("Attempting instantiation with positional arg...")
    meta = ToolMetadata(name="test", description="desc")
    t = BaseTool(meta)
    print("Success pos")
except Exception as e:
    print(f"Failed pos: {e}")
