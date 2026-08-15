# Phase 3 Prompt：注册集成 + ray-ascend 入口

你需要在 ray-ascend 项目中完成 HIXL tensor transport 的注册集成，确保 `register_hixl_tensor_transport()` 被调用后 `@ray.method(tensor_transport="HIXL")` 能正常工作。

## 背景

- `pyproject.toml` 缺少 `hixl` 可选依赖组

## 参考：现有注册集成模式

请严格参考以下现有代码的模式和风格：

1. **ray-ascend 注册入口风格**：参考 `ray-ascend/ray_ascend/__init__.py` 中已有的 `register_yr_tensor_transport()`（23-97 行）和 `register_hccl_tensor_transport()`（144-173 行）
   - 参数校验（devices is None → raise ValueError）
   - ImportError 处理（依赖不装 → 明确提示安装命令）
   - 最终调用 `register_tensor_transport(name, devices, Class, torch.Tensor)`

2. **pyproject.toml 可选依赖格式**：参考 `ray-ascend/pyproject.toml` 中已有的 `[project.optional-dependencies.yr]`（42-48 行）

## 需要完成的修改


### 修改 3：在 pyproject.toml 添加 hixl 可选依赖组

参考 `[project.optional-dependencies.yr]`（第 42-48 行）的格式，在 `ray-ascend/pyproject.toml` 的 `[project.optional-dependencies]` 中新增：

```toml
hixl = [
    "hixl_engine>=0.0.1",
    "torch>=2.7.1; platform_machine == 'x86_64'",
    "torch>=2.7.1; platform_machine == 'aarch64'",
    "torch-npu>=2.7.1.post2",
]
```

> 注意：`hixl_engine` 的实际包名和版本号需根据 wheel 包确认。如果 wheel 尚未发布到 PyPI，可暂用 URL 引用：`"hixl_engine @ https://<internal-url>/hixl_engine-0.0.1-py3-none-any.whl"`

### 修改 4：确认 __init__.py 注册入口

检查 `ray-ascend/ray_ascend/__init__.py` 中已有的 `register_hixl_tensor_transport()` 函数（176-246 行）是否与设计文档一致。设计文档在 `ray-ascend/docs/hixl-tensor-transport-design.md` 第 7 节。

重点确认：
1. `devices is None` 校验存在
2. 导入路径正确：`from ray_ascend.direct_transport.hixl_tensor_transport import HixlTensorTransport`
3. `hixl_wrapper` 可导入性检查存在
4. 最终调用 `register_tensor_transport("HIXL", devices, HixlTensorTransport, torch.Tensor)` — 这要求 `HixlTensorTransport` 必须是 `TensorTransportManager` 的子类

如果以上全部正确则无需改动；如有遗漏请修正。

## 输出要求

请修改以下文件：

1. `ray-ascend/pyproject.toml` — 添加 hixl 可选依赖组
2. `ray-ascend/ray_ascend/__init__.py` — 如需修正则修改（否则不动）