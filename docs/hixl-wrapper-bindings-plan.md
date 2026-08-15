# HIXL Engine Python 绑定实现计划

| 项目 | 内容 |
|---|---|
| 模块名 | `hixl_wrapper` |
| 目标 | 为 `hixl::Hixl` 类创建 pybind11 模块，供 Ray RDT 传输层调用 HIXL RDMA 传输能力 |
| 参考实现 | `hixl/src/python/llm_wrapper/`（LLM-DataDist Python 绑定） |
| 打包方式 | 独立 wheel（`hixl_engine-0.0.1-py3-none-any.whl`） |

---

## 1. 设计背景

### 1.1 架构层次

```
┌─────────────────────────────────────────────────────────────┐
│ LLM-DataDist（最高层）                                        │
│ 类: LlmDataDist / LLMDataDistV2                              │
│ 语义: KV Cache（RegisterCache、PullCache、TransferCache）     │
│ pybind: llm_datadist_wrapper ← 已存在，不需要改动             │
├─────────────────────────────────────────────────────────────┤
│ HIXL Engine（中间层）                                         │
│ 类: hixl::Hixl                                               │
│ 语义: 通用 RDMA 传输（RegisterMem、Connect、TransferSync）    │
│ pybind: hixl_wrapper ← 本计划要创建的                         │
├─────────────────────────────────────────────────────────────┤
│ ADXL（最底层）                                                │
│ 类: adxl::AdxlInnerEngine                                    │
│ 语义: RDMA/HCCS 硬件级传输                                    │
│ pybind: 无，不需要                                            │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 与参考实现（llm_wrapper）的关键差异

本设计严格遵循 `llm_wrapper` 的代码风格，但 HIXL Engine API 与 LLM-DataDist API 存在以下差异，需要特别注意：

| 差异点 | LLM-DataDist (llm_wrapper) | HIXL Engine (hixl_wrapper) | 处理方式 |
|---|---|---|---|
| 单例生命周期 | `Init(cluster_id, options)` → 创建并初始化 | `Initialize(local_engine, options)` → 创建并初始化 | 参数类型不同（`uint64_t cluster_id` vs `std::string local_engine`），但模式相同 |
| 资源清理 | `Finalize()` 返回 `void` | `Finalize()` 返回 `void` | **完全一致**，Python 端无返回值 |
| 输出参数数量 | 多个方法有 2 个输出参数（如 `RegisterCache` 返回 `Status + Cache`） | 大部分方法只有 0 或 1 个输出参数 | 仅 `RegisterMem`、`TransferAsync`、`GetTransferStatus`、`GetNotifies` 有输出参数 |
| `void*` handle | LLM-DataDist 不暴露 `void*` 到 Python | `MemHandle`/`TransferReq` 是 `void*`，需要 `uintptr_t` 桥接 | **新增类型映射**：`void*` ↔ `uintptr_t` ↔ Python `int` |
| 枚举类型 | LLM-DataDist 使用 `ge::Status` 等已有类型 | HIXL 使用自有枚举 `MemType`、`TransferOp`、`TransferStatus` | **新增 str ↔ enum 转换** |
| struct 拆包 | `CacheDesc`、`CacheKey` 等复杂 struct | `MemDesc`、`TransferOpDesc`、`NotifyDesc` 较简单 struct | Tuple 别名更简单（2-3 元素 vs 7-10 元素） |
| CMake 链接目标 | `llm_datadist` | `cann_hixl` | 链接目标不同 |
| Wheel 打包 | 合并到 `llm_datadist` wheel | **独立 wheel** | 新增独立打包流程 |

---

## 2. 文件清单

| 文件路径 | 作用 |
|---|---|
| `hixl/src/python/hixl_wrapper/hixl_engine_wrapper.h` | Wrapper 类声明（Tuple 类型别名 + 静态方法声明） |
| `hixl/src/python/hixl_wrapper/hixl_engine_wrapper.cc` | Wrapper 类实现（tuple↔C++ 转换 + 委托调用） |
| `hixl/src/python/hixl_wrapper/hixl_wrapper.cc` | pybind11 模块入口（注册函数和常量） |
| `hixl/src/python/hixl_wrapper/CMakeLists.txt` | 构建配置 |
| `hixl/src/python/hixl_engine/CMakeLists.txt` | 独立 wheel 打包配置 |
| `hixl/src/python/hixl_engine/setup.py` | wheel 打包脚本 |
| `hixl/src/python/hixl_engine/MANIFEST.in` | wheel 包含规则 |
| `hixl/src/python/hixl_engine/hixl_engine/__init__.py` | Python 包入口 |
| `hixl/src/python/CMakeLists.txt` | 新增 `add_subdirectory(hixl_wrapper)` 和 `add_subdirectory(hixl_engine)` |

---

## 3. C++ API 方法签名交叉验证

逐方法对比 `hixl::Hixl` C++ API（`include/hixl/hixl.h`）与 Wrapper 签名，确认映射正确性：

| # | C++ 方法 | C++ 签名 | Wrapper 签名 | 输出参数处理 | Python 返回值 | 验证结果 |
|---|---|---|---|---|---|---|
| 1 | `Initialize` | `Status Initialize(const AscendString &local_engine, const std::map<AscendString, AscendString> &options)` | `Status Initialize(const std::string &local_engine, const std::map<std::string, std::string> &options)` | 无输出参数 | `int` (status) | ✅ `AscendString` → `std::string`，`std::map` 直接映射 |
| 2 | `Finalize` | `void Finalize()` | `void Finalize()` | 无输出参数 | `None` | ✅ **与 C++ API 一致**，Python 端无返回值 |
| 3 | `RegisterMem` | `Status RegisterMem(const MemDesc &mem, MemType type, MemHandle &mem_handle)` | `std::pair<Status, uintptr_t> RegisterMem(const MemDescTuple &mem_desc, const std::string &mem_type)` | `MemHandle &mem_handle` → 第二个返回值 | `(int, int)` | ✅ `MemDesc` → `MemDescTuple`，`MemType` → `str`，`MemHandle` → `uintptr_t` |
| 4 | `DeregisterMem` | `Status DeregisterMem(MemHandle mem_handle)` | `Status DeregisterMem(uintptr_t mem_handle)` | 无输出参数 | `int` (status) | ✅ `MemHandle` → `uintptr_t` |
| 5 | `Connect` | `Status Connect(const AscendString &remote_engine, int32_t timeout_in_millis = 1000)` | `Status Connect(const std::string &remote_engine, int32_t timeout_ms = 1000)` | 无输出参数 | `int` (status) | ✅ 默认超时 1000ms 已保留 |
| 6 | `Disconnect` | `Status Disconnect(const AscendString &remote_engine, int32_t timeout_in_millis = 1000)` | `Status Disconnect(const std::string &remote_engine, int32_t timeout_ms = 1000)` | 无输出参数 | `int` (status) | ✅ 默认超时 1000ms 已保留 |
| 7 | `TransferSync` | `Status TransferSync(const AscendString &remote_engine, TransferOp operation, const std::vector<TransferOpDesc> &op_descs, int32_t timeout_in_millis = 1000)` | `Status TransferSync(const std::string &remote_engine, const std::string &operation, const std::vector<TransferOpDescTuple> &op_descs, int32_t timeout_ms = 1000)` | 无输出参数 | `int` (status) | ✅ `TransferOp` → `str`，默认超时 1000ms 已保留 |
| 8 | `TransferAsync` | `Status TransferAsync(const AscendString &remote_engine, TransferOp operation, const std::vector<TransferOpDesc> &op_descs, const TransferArgs &optional_args, TransferReq &req)` | `std::pair<Status, uintptr_t> TransferAsync(const std::string &remote_engine, const std::string &operation, const std::vector<TransferOpDescTuple> &op_descs)` | `TransferReq &req` → 第二个返回值；`TransferArgs` 内部构造 | `(int, int)` | ✅ `TransferArgs` 不暴露给 Python（reserved 字段默认为 0） |
| 9 | `GetTransferStatus` | `Status GetTransferStatus(const TransferReq &req, TransferStatus &status)` | `std::pair<Status, std::string> GetTransferStatus(uintptr_t req_id)` | `TransferStatus &status` → 第二个返回值（转为 str） | `(int, str)` | ✅ `TransferReq` → `uintptr_t`，`TransferStatus` → `str` |
| 10 | `SendNotify` | `Status SendNotify(const AscendString &remote_engine, const NotifyDesc &notify, int32_t timeout_in_millis = 1000)` | `Status SendNotify(const std::string &remote_engine, const NotifyDescTuple &notify, int32_t timeout_ms = 1000)` | 无输出参数 | `int` (status) | ✅ `NotifyDesc` → `NotifyDescTuple`，默认超时 1000ms 已保留 |
| 11 | `GetNotifies` | `Status GetNotifies(std::vector<NotifyDesc> &notifies)` | `std::pair<Status, std::vector<NotifyDescTuple>> GetNotifies()` | `std::vector<NotifyDesc> &notifies` → 第二个返回值（转为 tuple 列表） | `(int, list[tuple])` | ✅ |

**原设计文档发现的签名问题（已修正）：**

| 问题 | 原设计 | 修正 | 原因 |
|---|---|---|---|
| Finalize 返回值 | `std::tuple<Status>` → Python `(status,)` | `void` → Python `None` | C++ API 返回 `void`，与 llm_wrapper 参考实现一致 |
| 单返回值方法风格 | 全部用 `std::tuple<Status>` | 单返回值直接返回 `Status`；多返回值用 `std::pair<>` | 与 llm_wrapper 一致（`UnregisterCache` 返回 `ge::Status`，`RegisterCache` 返回 `std::pair<>`） |
| ParseMemType/ParseTransferOp | 非法字符串静默返回默认值 | 返回 `PARAM_INVALID` + ALOG 警告 | 避免掩盖调用错误 |
| GetNotifies 中 AscendString 转换 | 使用 `GetData()` | 使用 `GetString()` | `GetData()` 在本仓库的 stub 中不存在；`GetString()` 有 null 安全保证（返回 `""` 而不是 `nullptr`） |

---

## 4. 类型映射规则

| C++ 类型 | Python 类型 | C++ → Python | Python → C++ | 备注 |
|---|---|---|---|---|
| `AscendString` | `str` | `GetString()` → `std::string` | `std::string.c_str()` → `AscendString()` | 使用 `GetString()` 而非 `GetData()`（null 安全） |
| `MemHandle` (`void*`) | `int` | `reinterpret_cast<uintptr_t>(handle)` | `reinterpret_cast<MemHandle>(uintptr_t)` | Python 端持有的是地址整数，**底层释放后 Python 端的 int 变为悬空指针，调用方需自行管理生命周期** |
| `TransferReq` (`void*`) | `int` | `reinterpret_cast<uintptr_t>(req)` | `reinterpret_cast<TransferReq>(uintptr_t)` | 同 MemHandle，悬空指针风险 |
| `Status` (`uint32_t`) | `int` | 直接映射 | 直接映射 | |
| `MemType` (enum) | `str` | `"npu"` / `"cpu"` | `ParseMemType(str)` → enum | 非法字符串返回 `PARAM_INVALID` |
| `TransferOp` (enum) | `str` | `"READ"` / `"WRITE"` | `ParseTransferOp(str)` → enum | 非法字符串返回 `PARAM_INVALID` |
| `TransferStatus` (enum class) | `str` | `TransferStatusToStr()` | 不需要反向转换 | `"WAITING"` / `"COMPLETED"` / `"TIMEOUT"` / `"FAILED"` |
| `MemDesc` (struct) | `tuple(int, int)` | `UnpackMemDesc()` | `UnpackMemDesc()` | `(addr, len)` |
| `TransferOpDesc` (struct) | `tuple(int, int, int)` | `UnpackTransferOpDescs()` | `UnpackTransferOpDescs()` | `(local_addr, remote_addr, len)` |
| `NotifyDesc` (struct) | `tuple(str, str)` | `GetString()` → `std::string` | `UnpackNotifyDesc()` | `(name, msg)` |
| `TransferArgs` (struct) | 不暴露 | 内部构造 `{}` | — | `reserved[128] = {}` 默认为 0 |

---

## 5. 返回值设计

与 llm_wrapper 保持一致的风格：
- **单返回值方法**：直接返回 `Status`（Python 端收到 `int`）
- **多返回值方法**：返回 `std::pair<>`（Python 端收到 `tuple`）

| 方法 | Wrapper 返回类型 | Python 返回值 | 示例 |
|---|---|---|---|
| `Initialize` | `Status` | `int` | `status = hixl_wrapper.initialize(...)` |
| `Finalize` | `void` | `None` | `hixl_wrapper.finalize()` |
| `RegisterMem` | `std::pair<Status, uintptr_t>` | `(int, int)` | `status, handle = hixl_wrapper.register_mem(...)` |
| `DeregisterMem` | `Status` | `int` | `status = hixl_wrapper.deregister_mem(handle)` |
| `Connect` | `Status` | `int` | `status = hixl_wrapper.connect(...)` |
| `Disconnect` | `Status` | `int` | `status = hixl_wrapper.disconnect(...)` |
| `TransferSync` | `Status` | `int` | `status = hixl_wrapper.transfer_sync(...)` |
| `TransferAsync` | `std::pair<Status, uintptr_t>` | `(int, int)` | `status, req_id = hixl_wrapper.transfer_async(...)` |
| `GetTransferStatus` | `std::pair<Status, std::string>` | `(int, str)` | `status, ts = hixl_wrapper.get_transfer_status(req_id)` |
| `SendNotify` | `Status` | `int` | `status = hixl_wrapper.send_notify(...)` |
| `GetNotifies` | `std::pair<Status, std::vector<NotifyDescTuple>>` | `(int, list[tuple])` | `status, notifies = hixl_wrapper.get_notifies()` |

---

## 6. `hixl_engine_wrapper.h` — Wrapper 类声明

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * 请参阅 License 获取详细信息。
 */

#ifndef CANN_HIXL_PYTHON_HIXL_WRAPPER_HIXL_ENGINE_WRAPPER_H_
#define CANN_HIXL_PYTHON_HIXL_WRAPPER_HIXL_ENGINE_WRAPPER_H_

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "hixl/hixl.h"
#include "hixl/hixl_types.h"

namespace hixl_wrapper {

// Tuple 类型别名：Python ↔ C++ 的桥梁类型
using MemDescTuple        = std::tuple<uintptr_t, size_t>;              // (addr, length)
using TransferOpDescTuple = std::tuple<uintptr_t, uintptr_t, size_t>;  // (local_addr, remote_addr, len)
using NotifyDescTuple     = std::tuple<std::string, std::string>;      // (name, msg)

class HixlEngineWrapper {
 public:
  // 拆包方法：Python tuple → C++ struct
  static hixl::MemDesc UnpackMemDesc(const MemDescTuple &mem_desc_tuple);
  static std::vector<hixl::TransferOpDesc> UnpackTransferOpDescs(
      const std::vector<TransferOpDescTuple> &op_desc_tuples);
  static hixl::NotifyDesc UnpackNotifyDesc(const NotifyDescTuple &notify_tuple);

  // 字符串 ↔ 枚举转换（非法输入返回 PARAM_INVALID）
  static std::pair<hixl::Status, hixl::MemType> ParseMemType(const std::string &mem_type_str);
  static std::pair<hixl::Status, hixl::TransferOp> ParseTransferOp(const std::string &op_str);
  static std::string TransferStatusToStr(hixl::TransferStatus status);

  // 业务方法（全部 static，与 llm_wrapper 风格一致）
  static hixl::Status Initialize(const std::string &local_engine,
                                  const std::map<std::string, std::string> &options);
  static void Finalize();
  static std::pair<hixl::Status, uintptr_t> RegisterMem(const MemDescTuple &mem_desc,
                                                          const std::string &mem_type);
  static hixl::Status DeregisterMem(uintptr_t mem_handle);
  static hixl::Status Connect(const std::string &remote_engine, int32_t timeout_ms = 1000);
  static hixl::Status Disconnect(const std::string &remote_engine, int32_t timeout_ms = 1000);
  static hixl::Status TransferSync(const std::string &remote_engine,
                                    const std::string &operation,
                                    const std::vector<TransferOpDescTuple> &op_descs,
                                    int32_t timeout_ms = 1000);
  static std::pair<hixl::Status, uintptr_t> TransferAsync(const std::string &remote_engine,
                                                            const std::string &operation,
                                                            const std::vector<TransferOpDescTuple> &op_descs);
  static std::pair<hixl::Status, std::string> GetTransferStatus(uintptr_t req_id);
  static hixl::Status SendNotify(const std::string &remote_engine,
                                  const NotifyDescTuple &notify,
                                  int32_t timeout_ms = 1000);
  static std::pair<hixl::Status, std::vector<NotifyDescTuple>> GetNotifies();

 private:
  static std::unique_ptr<hixl::Hixl> hixl_engine_;
};

}  // namespace hixl_wrapper

#endif  // CANN_HIXL_PYTHON_HIXL_WRAPPER_HIXL_ENGINE_WRAPPER_H_
```

### 设计要点

- 所有方法都是 `static`——Python 端不需要创建实例，直接调用裸函数（与 llm_wrapper 一致）
- `hixl_engine_` 是 `static unique_ptr<Hixl>` 单例——`Initialize` 创建，`Finalize` 销毁（与 llm_wrapper 一致）
- `MemHandle`/`TransferReq`（都是 `void*`）用 `uintptr_t` 传递给 Python（Python `int`），调用方需自行管理生命周期，避免悬空指针
- 枚举类型在 Python 端用 `str` 表示（"npu"/"cpu"，"READ"/"WRITE"，"WAITING"/"COMPLETED" 等）
- `ParseMemType`/`ParseTransferOp` 返回 `std::pair<Status, enum>`——非法字符串返回 `PARAM_INVALID` + ALOG 警告，不再静默吞错
- 带默认超时参数的方法（`Connect`/`Disconnect`/`TransferSync`/`SendNotify`）默认值为 1000ms，与 C++ API 一致
- `Finalize` 返回 `void`——与 C++ API 一致，Python 端无返回值
- 单返回值方法直接返回 `Status`（Python `int`），多返回值方法返回 `std::pair<>`（Python `tuple`）——与 llm_wrapper 一致

---

## 7. `hixl_engine_wrapper.cc` — Wrapper 类实现

### 7.1 生命周期管理

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * 请参阅 License 获取详细信息。
 */

#include "hixl_engine_wrapper.h"
#include "alog/alog.h"

namespace hixl_wrapper {

std::unique_ptr<hixl::Hixl> HixlEngineWrapper::hixl_engine_;

hixl::Status HixlEngineWrapper::Initialize(
    const std::string &local_engine,
    const std::map<std::string, std::string> &options) {
  if (hixl_engine_ != nullptr) {
    ALOG_WARN("HixlEngineWrapper::Initialize: engine already initialized, repeat init");
    return hixl::PARAM_INVALID;
  }
  hixl_engine_ = std::make_unique<hixl::Hixl>();
  hixl::AscendString ascend_local_engine(local_engine.c_str());
  std::map<hixl::AscendString, hixl::AscendString> ascend_options;
  for (const auto &opt : options) {
    ascend_options.emplace(hixl::AscendString(opt.first.c_str()),
                           hixl::AscendString(opt.second.c_str()));
  }
  auto ret = hixl_engine_->Initialize(ascend_local_engine, ascend_options);
  if (ret != hixl::SUCCESS) {
    ALOG_ERROR("HixlEngineWrapper::Initialize: failed, ret=%u", ret);
    hixl_engine_.reset();
  }
  return ret;
}

void HixlEngineWrapper::Finalize() {
  if (hixl_engine_ != nullptr) {
    hixl_engine_->Finalize();
    hixl_engine_.reset();
  }
}
```

> **注意**：`Finalize` 返回 `void`，与 C++ API（`void Hixl::Finalize()`）和 llm_wrapper 参考实现（`void LLMDataDistV2Wrapper::Finalize()`）完全一致。Python 端调用 `hixl_wrapper.finalize()` 无返回值。

### 7.2 字符串 ↔ 枚举转换

```cpp
std::pair<hixl::Status, hixl::MemType> HixlEngineWrapper::ParseMemType(const std::string &mem_type_str) {
  if (mem_type_str == "npu") return {hixl::SUCCESS, hixl::MEM_DEVICE};
  if (mem_type_str == "cpu") return {hixl::SUCCESS, hixl::MEM_HOST};
  ALOG_WARN("HixlEngineWrapper::ParseMemType: invalid mem_type '%s', expected 'npu' or 'cpu'",
            mem_type_str.c_str());
  return {hixl::PARAM_INVALID, hixl::MEM_DEVICE};
}

std::pair<hixl::Status, hixl::TransferOp> HixlEngineWrapper::ParseTransferOp(const std::string &op_str) {
  if (op_str == "READ")  return {hixl::SUCCESS, hixl::READ};
  if (op_str == "WRITE") return {hixl::SUCCESS, hixl::WRITE};
  ALOG_WARN("HixlEngineWrapper::ParseTransferOp: invalid operation '%s', expected 'READ' or 'WRITE'",
            op_str.c_str());
  return {hixl::PARAM_INVALID, hixl::READ};
}

std::string HixlEngineWrapper::TransferStatusToStr(hixl::TransferStatus status) {
  switch (status) {
    case hixl::TransferStatus::WAITING:   return "WAITING";
    case hixl::TransferStatus::COMPLETED: return "COMPLETED";
    case hixl::TransferStatus::TIMEOUT:   return "TIMEOUT";
    case hixl::TransferStatus::FAILED:    return "FAILED";
    default:                              return "UNKNOWN";
  }
}
```

> **设计决策**：`ParseMemType`/`ParseTransferOp` 返回 `std::pair<Status, enum>`，非法字符串返回 `PARAM_INVALID` + ALOG 警告。调用方需检查 Status 后再使用 enum 值。这避免了原设计中非法字符串被静默吞错的问题。

### 7.3 拆包方法

```cpp
hixl::MemDesc HixlEngineWrapper::UnpackMemDesc(const MemDescTuple &t) {
  hixl::MemDesc mem_desc{};
  mem_desc.addr = std::get<0>(t);
  mem_desc.len  = std::get<1>(t);
  // reserved[128] 默认值为 0，MemDesc 定义已有 = {}
  return mem_desc;
}

std::vector<hixl::TransferOpDesc> HixlEngineWrapper::UnpackTransferOpDescs(
    const std::vector<TransferOpDescTuple> &op_desc_tuples) {
  std::vector<hixl::TransferOpDesc> op_descs;
  op_descs.reserve(op_desc_tuples.size());
  for (const auto &t : op_desc_tuples) {
    hixl::TransferOpDesc desc{};
    desc.local_addr  = std::get<0>(t);
    desc.remote_addr = std::get<1>(t);
    desc.len         = std::get<2>(t);
    op_descs.emplace_back(desc);
  }
  return op_descs;
}

hixl::NotifyDesc HixlEngineWrapper::UnpackNotifyDesc(const NotifyDescTuple &t) {
  hixl::NotifyDesc notify{};
  notify.name       = hixl::AscendString(std::get<0>(t).c_str());
  notify.notify_msg = hixl::AscendString(std::get<1>(t).c_str());
  return notify;
}
```

### 7.4 RegisterMem — 有输出参数的方法

```cpp
std::pair<hixl::Status, uintptr_t> HixlEngineWrapper::RegisterMem(
    const MemDescTuple &mem_desc_tuple, const std::string &mem_type_str) {
  hixl::MemHandle handle = nullptr;
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    auto [parse_status, mem_type] = ParseMemType(mem_type_str);
    if (parse_status != hixl::SUCCESS) {
      return {parse_status, reinterpret_cast<uintptr_t>(handle)};
    }
    auto mem_desc = UnpackMemDesc(mem_desc_tuple);
    ret = hixl_engine_->RegisterMem(mem_desc, mem_type, handle);
  } else {
    ALOG_WARN("HixlEngineWrapper::RegisterMem: engine not initialized");
  }
  return {ret, reinterpret_cast<uintptr_t>(handle)};
}
```

### 7.5 DeregisterMem

```cpp
hixl::Status HixlEngineWrapper::DeregisterMem(uintptr_t mem_handle) {
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    hixl::MemHandle handle = reinterpret_cast<hixl::MemHandle>(mem_handle);
    ret = hixl_engine_->DeregisterMem(handle);
  } else {
    ALOG_WARN("HixlEngineWrapper::DeregisterMem: engine not initialized");
  }
  return ret;
}
```

### 7.6 Connect

```cpp
hixl::Status HixlEngineWrapper::Connect(const std::string &remote_engine, int32_t timeout_ms) {
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    hixl::AscendString ascend_remote(remote_engine.c_str());
    ret = hixl_engine_->Connect(ascend_remote, timeout_ms);
  } else {
    ALOG_WARN("HixlEngineWrapper::Connect: engine not initialized");
  }
  return ret;
}
```

> **默认超时**：`timeout_ms` 默认值为 1000ms，与 C++ API `int32_t timeout_in_millis = 1000` 一致。Python 端调用 `hixl_wrapper.connect("IP:PORT")` 不传 timeout 时使用 1000ms。

### 7.7 Disconnect

```cpp
hixl::Status HixlEngineWrapper::Disconnect(const std::string &remote_engine, int32_t timeout_ms) {
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    hixl::AscendString ascend_remote(remote_engine.c_str());
    ret = hixl_engine_->Disconnect(ascend_remote, timeout_ms);
  } else {
    ALOG_WARN("HixlEngineWrapper::Disconnect: engine not initialized");
  }
  return ret;
}
```

### 7.8 TransferSync

```cpp
hixl::Status HixlEngineWrapper::TransferSync(
    const std::string &remote_engine,
    const std::string &operation,
    const std::vector<TransferOpDescTuple> &op_desc_tuples,
    int32_t timeout_ms) {
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    auto [parse_status, op] = ParseTransferOp(operation);
    if (parse_status != hixl::SUCCESS) {
      return parse_status;
    }
    hixl::AscendString ascend_remote(remote_engine.c_str());
    auto op_descs = UnpackTransferOpDescs(op_desc_tuples);
    ret = hixl_engine_->TransferSync(ascend_remote, op, op_descs, timeout_ms);
  } else {
    ALOG_WARN("HixlEngineWrapper::TransferSync: engine not initialized");
  }
  return ret;
}
```

### 7.9 TransferAsync — 有输出参数的方法

```cpp
std::pair<hixl::Status, uintptr_t> HixlEngineWrapper::TransferAsync(
    const std::string &remote_engine,
    const std::string &operation,
    const std::vector<TransferOpDescTuple> &op_desc_tuples) {
  hixl::TransferReq req = nullptr;
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    auto [parse_status, op] = ParseTransferOp(operation);
    if (parse_status != hixl::SUCCESS) {
      return {parse_status, reinterpret_cast<uintptr_t>(req)};
    }
    hixl::AscendString ascend_remote(remote_engine.c_str());
    auto op_descs = UnpackTransferOpDescs(op_desc_tuples);
    hixl::TransferArgs args{};  // reserved[128] 默认为 0
    ret = hixl_engine_->TransferAsync(ascend_remote, op, op_descs, args, req);
  } else {
    ALOG_WARN("HixlEngineWrapper::TransferAsync: engine not initialized");
  }
  return {ret, reinterpret_cast<uintptr_t>(req)};
}
```

> **悬空指针风险**：`TransferAsync` 返回的 `req_id`（`uintptr_t`）指向底层引擎的内部数据。如果引擎被 `Finalize` 销毁后再用此 `req_id` 调用 `GetTransferStatus`，会导致悬空指针访问。**调用方需确保在 Finalize 前完成所有异步传输查询。**

### 7.10 GetTransferStatus — 有输出参数的方法

```cpp
std::pair<hixl::Status, std::string> HixlEngineWrapper::GetTransferStatus(uintptr_t req_id) {
  hixl::TransferStatus status = hixl::TransferStatus::FAILED;
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    hixl::TransferReq req = reinterpret_cast<hixl::TransferReq>(req_id);
    ret = hixl_engine_->GetTransferStatus(req, status);
  } else {
    ALOG_WARN("HixlEngineWrapper::GetTransferStatus: engine not initialized");
  }
  return {ret, TransferStatusToStr(status)};
}
```

### 7.11 SendNotify

```cpp
hixl::Status HixlEngineWrapper::SendNotify(
    const std::string &remote_engine,
    const NotifyDescTuple &notify_tuple,
    int32_t timeout_ms) {
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    hixl::AscendString ascend_remote(remote_engine.c_str());
    auto notify = UnpackNotifyDesc(notify_tuple);
    ret = hixl_engine_->SendNotify(ascend_remote, notify, timeout_ms);
  } else {
    ALOG_WARN("HixlEngineWrapper::SendNotify: engine not initialized");
  }
  return ret;
}
```

### 7.12 GetNotifies — 有输出参数的方法

```cpp
std::pair<hixl::Status, std::vector<NotifyDescTuple>> HixlEngineWrapper::GetNotifies() {
  std::vector<hixl::NotifyDesc> notifies;
  hixl::Status ret = hixl::FAILED;
  if (hixl_engine_ != nullptr) {
    ret = hixl_engine_->GetNotifies(notifies);
  } else {
    ALOG_WARN("HixlEngineWrapper::GetNotifies: engine not initialized");
  }
  std::vector<NotifyDescTuple> notify_tuples;
  for (const auto &n : notifies) {
    // 使用 GetString() 而非 GetData()：
    // GetString() 有 null 安全保证（name_ 为 nullptr 时返回 ""），GetData() 可能返回 nullptr 导致 crash
    std::string name(n.name.GetString());
    std::string msg(n.notify_msg.GetString());
    notify_tuples.emplace_back(std::make_tuple(name, msg));
  }
  return {ret, notify_tuples};
}

}  // namespace hixl_wrapper
```

> **设计决策**：使用 `GetString()` 而非 `GetData()`。原因：
> 1. 本仓库的 AscendString stub 实现中只有 `GetString()` 方法，没有 `GetData()` 方法
> 2. `GetString()` 有 null 安全保证：当内部 `name_` 为 nullptr 时返回静态空字符串 `""`，而 `GetData()`（如果 CANN SDK 中存在）可能返回 `nullptr`，传给 `std::string(nullptr)` 会 crash

---

## 8. `hixl_wrapper.cc` — pybind11 模块入口

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * 请参阅 License 获取详细信息。
 */

#include "Python.h"
#ifdef ASCEND_CI_LIMITED_PY37
#undef PyCFunction_NewEx
#endif

#include <map>
#include <string>
#include <vector>
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "hixl/hixl.h"
#include "hixl/hixl_types.h"
#include "hixl_engine_wrapper.h"

#undef PYBIND11_CHECK_PYTHON_VERSION
#define PYBIND11_CHECK_PYTHON_VERSION

namespace hixl_wrapper {
namespace {
namespace py = pybind11;

void BindStatusCodes(py::module &m) {
  // Status 状态码（与 llm_wrapper 的 BindStatusCodes 风格一致）
  m.attr("kSuccess")           = py::int_(hixl::SUCCESS);
  m.attr("kFailed")            = py::int_(hixl::FAILED);
  m.attr("kParamInvalid")      = py::int_(hixl::PARAM_INVALID);
  m.attr("kTimeout")           = py::int_(hixl::TIMEOUT);
  m.attr("kNotConnected")      = py::int_(hixl::NOT_CONNECTED);
  m.attr("kAlreadyConnected")  = py::int_(hixl::ALREADY_CONNECTED);
  m.attr("kNotifyFailed")      = py::int_(hixl::NOTIFY_FAILED);
  m.attr("kUnsupported")       = py::int_(hixl::UNSUPPORTED);
  m.attr("kResourceExhausted") = py::int_(hixl::RESOURCE_EXHAUSTED);

  // MemType 枚举值（供 Python 端可选使用，虽然主要用 str 传参）
  m.attr("kMemDevice") = py::int_(hixl::MEM_DEVICE);
  m.attr("kMemHost")   = py::int_(hixl::MEM_HOST);

  // TransferOp 枚举值
  m.attr("kRead")  = py::int_(hixl::READ);
  m.attr("kWrite") = py::int_(hixl::WRITE);

  // 初始化选项常量（与 C++ OPTION_* 一致）
  m.attr("kOptionEnableUseFabricMem")    = py::str(hixl::OPTION_ENABLE_USE_FABRIC_MEM);
  m.attr("kOptionRdmaTrafficClass")      = py::str(hixl::OPTION_RDMA_TRAFFIC_CLASS);
  m.attr("kOptionRdmaServiceLevel")      = py::str(hixl::OPTION_RDMA_SERVICE_LEVEL);
  m.attr("kOptionBufferPool")            = py::str(hixl::OPTION_BUFFER_POOL);
  m.attr("kOptionGlobalResourceConfig")  = py::str(hixl::OPTION_GLOBAL_RESOURCE_CONFIG);
}

void BuildHixlFuncs(py::module &m) {
  // 所有方法使用 py::call_guard<py::gil_scoped_release>()：
  // C++ 操作不访问 Python 对象，释放 GIL 让其他 Python 线程并发执行
  (void)m.def("initialize",          &HixlEngineWrapper::Initialize,          py::call_guard<py::gil_scoped_release>());
  (void)m.def("finalize",            &HixlEngineWrapper::Finalize,            py::call_guard<py::gil_scoped_release>());
  (void)m.def("register_mem",        &HixlEngineWrapper::RegisterMem,         py::call_guard<py::gil_scoped_release>());
  (void)m.def("deregister_mem",      &HixlEngineWrapper::DeregisterMem,       py::call_guard<py::gil_scoped_release>());
  (void)m.def("connect",             &HixlEngineWrapper::Connect,             py::call_guard<py::gil_scoped_release>());
  (void)m.def("disconnect",          &HixlEngineWrapper::Disconnect,          py::call_guard<py::gil_scoped_release>());
  (void)m.def("transfer_sync",       &HixlEngineWrapper::TransferSync,        py::call_guard<py::gil_scoped_release>());
  (void)m.def("transfer_async",      &HixlEngineWrapper::TransferAsync,       py::call_guard<py::gil_scoped_release>());
  (void)m.def("get_transfer_status", &HixlEngineWrapper::GetTransferStatus,   py::call_guard<py::gil_scoped_release>());
  (void)m.def("send_notify",         &HixlEngineWrapper::SendNotify,          py::call_guard<py::gil_scoped_release>());
  (void)m.def("get_notifies",        &HixlEngineWrapper::GetNotifies,         py::call_guard<py::gil_scoped_release>());
}

}  // namespace

PYBIND11_MODULE(hixl_wrapper, m) {
  BindStatusCodes(m);
  BuildHixlFuncs(m);
}

}  // namespace hixl_wrapper
```

> **兼容性处理**（与 llm_wrapper 一致）：
> - `#include "Python.h"` + `ASCEND_CI_LIMITED_PY37` 宏：处理 Python 3.7 兼容性
> - `PYBIND11_CHECK_PYTHON_VERSION`：允许在构建时 Python 版本与运行时不完全匹配（CANN 构建环境可能使用不同 Python 版本）

---

## 9. GIL 释放策略

所有方法使用 `py::call_guard<py::gil_scoped_release>()`，理由：

| 方法 | 释放 GIL 的理由 |
|---|---|
| `Initialize` | 涉及硬件资源初始化，可能耗时 |
| `Finalize` | 涉及硬件资源销毁，可能耗时 |
| `RegisterMem` | 涉及 NPU 内存注册，可能涉及 RDMA 操作 |
| `DeregisterMem` | 涉及内存解注册 |
| `Connect` | 涉及网络建链，需要等待远端响应 |
| `Disconnect` | 涉及网络断链 |
| `TransferSync` | 同步传输，等待硬件完成 |
| `TransferAsync` | 下发传输请求，涉及硬件操作 |
| `GetTransferStatus` | 查询传输状态 |
| `SendNotify` | 涉及网络通信 |
| `GetNotifies` | 查询通知信息 |

所有 C++ 操作都不访问 Python 对象，释放 GIL 可以让其他 Python 线程并发执行。

> **注意**：`Finalize` 返回 `void`，pybind11 对 `void` 返回值自动返回 `None` 到 Python。GIL 释放不影响返回值传递。

---

## 10. `CMakeLists.txt` — 构建配置

```cmake
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# 请参阅 License 获取详细信息。
# ----------------------------------------------------------------------------

if (NOT ENABLE_TEST)
    set(CMAKE_SKIP_RPATH TRUE)
    add_library(hixl_wrapper MODULE
            hixl_wrapper.cc
            hixl_engine_wrapper.cc)

    target_include_directories(hixl_wrapper PRIVATE
            ${HI_PYTHON_INC}
            ${pybind11_INCLUDE_DIR}
            ${HIXL_CODE_DIR}/include
            ${HIXL_CODE_DIR}/src/llm_datadist
    )

    target_link_libraries(hixl_wrapper PRIVATE
            $<BUILD_INTERFACE:intf_pub>
            $<BUILD_INTERFACE:slog_headers>
            $<BUILD_INTERFACE:mmpa_headers>
            $<BUILD_INTERFACE:hccl_headers>
            $<BUILD_INTERFACE:metadef_headers>
            $<BUILD_INTERFACE:msprof_headers>
            alog
            cann_hixl
    )

    set_target_properties(hixl_wrapper
            PROPERTIES
            PREFIX ""
    )

    target_compile_definitions(hixl_wrapper PRIVATE
            PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF
    )

    target_compile_options(hixl_wrapper PRIVATE
            -Xlinker -export-dynamic
    )

    target_link_options(hixl_wrapper PRIVATE
            -s
    )
endif ()
```

> **与 `llm_wrapper` CMakeLists.txt 的对比**：
>
> | 配置项 | llm_wrapper | hixl_wrapper | 说明 |
> |---|---|---|---|
> | 链接目标 | `llm_datadist` | `cann_hixl` | 绑定不同层的 C++ 库 |
> | 源文件 | `llm_wrapper_v2.cc` + `llm_datadist_v2_wrapper.cc` | `hixl_wrapper.cc` + `hixl_engine_wrapper.cc` | |
> | include 目录 | 相同 | 相同（需 `src/llm_datadist` 因为 `ge_api_error_codes.h` 通过 `metadef_headers` 提供） | |
> | 其他配置 | 相同 | 相同 | `PREFIX ""`, `PYBIND11_NO_ASSERT_GIL_HELD_INCREF_DECREF`, `-Xlinker -export-dynamic`, `-s` |

---

## 11. 更新 `hixl/src/python/CMakeLists.txt`

```cmake
add_subdirectory(llm_datadist)
add_subdirectory(llm_wrapper)
add_subdirectory(metadef_wrapper)
add_subdirectory(hixl_wrapper)     # ← 新增：HIXL Engine Python 绑定
add_subdirectory(hixl_engine)      # ← 新增：独立 wheel 打包
```

---

## 12. 独立 Wheel 打包

### 12.1 目录结构

```
hixl/src/python/hixl_engine/
├── CMakeLists.txt        # wheel 打包构建配置
├── setup.py              # setuptools 打包脚本
├── MANIFEST.in           # 包含 .so 文件的规则
└── hixl_engine/
    └── __init__.py       # Python 包入口
```

### 12.2 `hixl_engine/__init__.py`

```python
"""HIXL Engine Python binding package."""
from hixl_engine import hixl_wrapper  # noqa: F401
```

### 12.3 `setup.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# 请参阅 License 获取详细信息。
# ----------------------------------------------------------------------------

from setuptools import setup, find_packages

setup(
    name='hixl_engine',
    version='0.0.1',
    description='hixl engine api',
    packages=find_packages(),
    include_package_data=True,
    ext_modules=[]
)
```

### 12.4 `MANIFEST.in`

```
recursive-include * *.so
```

### 12.5 `CMakeLists.txt` — wheel 打包

```cmake
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# 请参阅 License 获取详细信息。
# ----------------------------------------------------------------------------

add_custom_target(hixl_engine_python ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/hixl_engine-0.0.1-py3-none-any.whl)
add_custom_command(
        OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/hixl_engine-0.0.1-py3-none-any.whl
        COMMAND echo "package hixl engine whl start"
        && mkdir -p ${CMAKE_CURRENT_BINARY_DIR}/wheel1
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/setup.py ${CMAKE_CURRENT_BINARY_DIR}/wheel1/setup.py
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/MANIFEST.in ${CMAKE_CURRENT_BINARY_DIR}/wheel1/
        && cp -r ${CMAKE_CURRENT_SOURCE_DIR}/hixl_engine ${CMAKE_CURRENT_BINARY_DIR}/wheel1/
        && cp -r ${CMAKE_CURRENT_BINARY_DIR}/../hixl_wrapper/hixl_wrapper.so ${CMAKE_CURRENT_BINARY_DIR}/wheel1/hixl_engine/
        && cd ${CMAKE_CURRENT_BINARY_DIR}/wheel1
        && ${HI_PYTHON} setup.py bdist_wheel >/dev/null
        && cp -f dist/hixl_engine-0.0.1-py3-none-any.whl ${CMAKE_CURRENT_BINARY_DIR}/
        && echo "package hixl engine whl end"
        DEPENDS hixl_wrapper
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/hixl_engine-0.0.1-py3-none-any.whl OPTIONAL
        DESTINATION ${INSTALL_LIBRARY_DIR}
)
```

> **与 `llm_datadist` wheel 打包的对比**：
>
> | 配置项 | llm_datadist | hixl_engine | 说明 |
> |---|---|---|---|
> | wheel 名 | `llm_datadist-0.0.1-py3-none-any.whl` | `hixl_engine-0.0.1-py3-none-any.whl` | 独立 wheel |
> | 包含 .so | `llm_datadist_wrapper.so` + `metadef_wrapper.so` | `hixl_wrapper.so` | 只包含一个 .so |
> | Python 包名 | `llm_datadist` | `hixl_engine` | |
> | DEPENDS | `llm_datadist_wrapper` + `metadef_wrapper` + `generate_hixl_version_info` | `hixl_wrapper` | |
> | Python 导入 | `from llm_datadist import llm_datadist_wrapper` | `from hixl_engine import hixl_wrapper` | |

---

## 13. Python 端使用示例

```python
import hixl_wrapper

# 初始化引擎
status = hixl_wrapper.initialize("192.168.1.1:5000", {hixl_wrapper.kOptionBufferPool: "4G"})
if status != hixl_wrapper.kSuccess:
    raise RuntimeError(f"Initialize failed, status={status}")

# 注册内存（NPU 设备内存）
status, handle = hixl_wrapper.register_mem((0x1000, 4096), "npu")
if status != hixl_wrapper.kSuccess:
    raise RuntimeError(f"RegisterMem failed, status={status}")

# 连接远端（默认超时 1000ms）
status = hixl_wrapper.connect("192.168.1.2:5000")
# 或显式指定超时：
status = hixl_wrapper.connect("192.168.1.2:5000", 3000)

# 同步传输（READ：从远端拉到本地，默认超时 1000ms）
status = hixl_wrapper.transfer_sync(
    "192.168.1.2:5000",
    "READ",
    [(local_addr, remote_addr, length)],
    5000  # 显式指定超时
)

# 异步传输（WRITE：将本地写到远端）
status, req_id = hixl_wrapper.transfer_async(
    "192.168.1.2:5000",
    "WRITE",
    [(local_addr, remote_addr, length)]
)

# 查询传输状态（轮询直到 COMPLETED）
status, transfer_status = hixl_wrapper.get_transfer_status(req_id)
# transfer_status 可能是 "WAITING", "COMPLETED", "TIMEOUT", "FAILED"

# 发送通知（默认超时 1000ms）
status = hixl_wrapper.send_notify(
    "192.168.1.2:5000",
    ("signal_name", "message_content"),
    3000
)

# 获取通知
status, notifies = hixl_wrapper.get_notifies()
for name, msg in notifies:
    print(f"Notify: {name} - {msg}")

# 清理（必须在 Finalize 前完成所有异步传输查询）
status = hixl_wrapper.deregister_mem(handle)
status = hixl_wrapper.disconnect("192.168.1.2:5000")
hixl_wrapper.finalize()  # 返回 None，无返回值
```

---

## 14. 安全注意事项

### 14.1 悬空指针风险

`MemHandle` 和 `TransferReq` 都是 `void*`，通过 `uintptr_t` 桥接传递给 Python。Python 端持有的是地址整数。**如果底层引擎释放了 handle/req，Python 端还持有该整数并再次传回 C++，会导致悬空指针访问**。

**调用方需遵守的生命周期规则**：
1. `DeregisterMem(handle)` 后，不可再用该 `handle` 调用任何方法
2. 异步传输完成后（`GetTransferStatus` 返回 `"COMPLETED"`），`req_id` 不再有效
3. `Finalize()` 前必须完成所有异步传输查询和资源释放

### 14.2 枚举字符串校验

`ParseMemType`/`ParseTransferOp` 对非法字符串返回 `PARAM_INVALID` + ALOG 警告，不再静默返回默认值。调用方需检查 Status 后再使用结果。

### 14.3 AscendString null 安全

`GetNotifies` 中使用 `GetString()` 而非 `GetData()` 转换 `AscendString` → `std::string`。`GetString()` 在 `AscendString` 内部为 null 时返回 `""`（空字符串），避免 `std::string(nullptr)` crash。