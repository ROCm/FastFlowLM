---
layout: docs
title: Get Started (Linux)
nav_order: 2
has_children: false
---

# Linux NPU Support

This article will teach you how to run LLMs on your **AMD XDNA 2 NPU** on Linux using **FastFlowLM**.  
Get set up and then show us what you build!

**Date:** March 5, 2026  
**Authors:** [Lemonade-server🍋](https://lemonade-server.ai/) and FastFlowLM contributors

## 📢 FastFlowLM Linux Support

[FastFlowLM](https://github.com/FastFlowLM/FastFlowLM) is a lightweight LLM runtime optimized for **AMD NPUs**.  
Today, FastFlowLM is adding support for **Ubuntu, Arch, and other distros** to enable **fast, low-power LLMs** on **Ryzen™ AI PCs that run Linux**.

This article will help you:

- Understand **Linux NPU support status** and required platform versions
- Install the **FLM + driver stack** for your distribution
- Validate your setup with `flm validate`
- Fix common **firmware, driver, and memlock issues**

---

## ⚙️ Hardware Requirements

### Supported processors

FastFlowLM on Linux requires an **AMD XDNA 2 NPU**.

| Ryzen AI family | Codename | Status |
|---|---|---|
| Max 300-series | Strix Halo | Supported |
| 300-series | Kraken Point, Strix Point | Supported |
| 400-series | Gorgon Point | Supported |
| Z2 Extreme | Handheld devices | Supported |

> **Note:** Ryzen AI 7000 / 8000 / 200-series chips have **XDNA 1**, which is **not supported**.

---

## 🧰 Software Requirements

### Runtime stack

The NPU requires specific firmware, kernel version, driver, and runtime software to function.  
The quickstart guide below will help you install these requirements.

| Item | Requirement |
|---|---|
| NPU firmware | Version 1.1.0.0 or later |
| Kernel + driver | Kernel **7.0+** with `amdxdna`, or `amdxdna-dkms` |
| Runtime | FastFlowLM installed |
| Memlock limit | Must be high enough for NPU execution |

---

## 🚀 Quickstart

## Supported Distributions
- Ubuntu 24.04 LTS
- Ubuntu 25.10
- Ubuntu 26.04
- Arch Linux
- Other (Generic Linux)

---

## 1. Prerequisites
- `amdxdna` driver (included in kernel 7.0+, or via `amdxdna-dkms`)
- NPU firmware version 1.1.0.0 or later
- Python 3.8+
- XRT stack from AMD

---

## 2. System Preparation

### Ubuntu (24.04, 25.10)

#### 1. Add the AMD XRT PPA (Required for NPU/XDNA)
The AMD XRT stack is a prerequisite for NPU support. Add AMD's PPA:
```sh
sudo add-apt-repository ppa:lemonade-team/stable
sudo apt update
```
See [lemonade-team/stable PPA](https://launchpad.net/~lemonade-team/+archive/ubuntu/stable) for details.

#### 2. Install XRT and NPU Drivers
```sh
sudo apt install libxrt-npu2 amdxdna-dkms
```

#### 3. Reboot
```sh
sudo reboot
```

#### 4. Install FastFlowLM
- Download the latest `.deb` package from the [Releases page](https://github.com/FastFlowLM/FastFlowLM/releases):

```sh
sudo apt install ./fastflowlm*.deb
```

#### 5. (NPU) Check memlock limit
- Run:
   ```sh
   ulimit -l
   ```
- If not `unlimited`, add to `/etc/security/limits.conf`:

   `*    soft    memlock    unlimited`   
   `*    hard    memlock    unlimited`
- Reboot system

---

### Ubuntu 26.04, Arch, and Others

Check this [https://lemonade-server.ai/flm_npu_linux.html](https://lemonade-server.ai/flm_npu_linux.html)

---

### Building from Source

1. Clone the repository:
   ```sh
   git clone https://github.com/FastFlowLM/FastFlowLM.git
   cd FastFlowLM
   ```
2. Build:
   ```sh
   cd src
   cmake --preset linux-default
   cmake --build --preset linux-default -j$(nproc)
   cmake --install --preset linux-default
   ```

---

## 4. Validating NPU Setup

To validate your NPU setup, run:
```sh
flm validate
```
You should see output similar to:
```
[Linux]  Kernel: 7.0.0-rc1-00052-g27936bfca73d
[Linux]  NPU: /dev/accel/accel0
[Linux]  NPU FW Version: 1.1.2.64
[Linux]  Memlock Limit: infinity
```

---

## 📚 Additional resources

- [Lemonade-server🍋](https://lemonade-server.ai/)
- [Lemonade GitHub issues](https://github.com/lemonade-ai/lemonade/issues)
- [Lemonade Discord](https://discord.gg)

---