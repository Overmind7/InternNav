<div align="center">

[![demo](assets/InternNav.gif "demo")](https://www.youtube.com/watch?v=fD0F1jIax5Y)

[![HomePage](https://img.shields.io/badge/HomePage-144B9E?logo=ReactOS&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/)
[![Technique Report](https://img.shields.io/badge/Paper-B31B1B?logo=arXiv&logoColor=white)](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf)
[![doc](https://img.shields.io/badge/Document-FFA500?logo=readthedocs&logoColor=white)](https://internrobotics.github.io/user_guide/internnav/index.html)
[![GitHub star chart](https://img.shields.io/github/stars/InternRobotics/InternNav?style=square)](https://github.com/InternRobotics/InternNav)
[![GitHub Issues](https://img.shields.io/github/issues/InternRobotics/InternNav)](https://github.com/InternRobotics/InternNav/issues)
<a href="https://cdn.vansin.top/taoyuan.jpg"><img src="https://img.shields.io/badge/WeChat-07C160?logo=wechat&logoColor=white" height="20" style="display:inline"></a>
[![Discord](https://img.shields.io/discord/1373946774439591996?logo=discord)](https://discord.gg/5jeaQHUj4B)

</div>

## 🏠 介绍

InternNav 是基于 PyTorch、Habitat 与 Isaac Sim 的开源体态导航全流程工具箱。

### 亮点
- **完整导航系统的模块化支持**

  覆盖导航系统的各个环节，可自由组合与研究离散动作空间的视觉-语言导航（VLN-CE）、基于点/图像/轨迹目标的视觉导航（VN），以及输出连续轨迹的端到端 VLN 系统。

- **主流仿真平台兼容**

  适配不同的训练与评估需求，支持 Habitat、Isaac Sim 等主流仿真平台及其环境配置。

- **丰富的数据集、模型与基准**

  支持 6 套最完整的数据集与基准、10+ 流行基线，涵盖主流与团队自建的方案。

- **SOTA 真实表现**

  内置高质量导航数据集 InternData-N1（3k+ 场景、830k VLN 数据，覆盖多种机体与场景），以及首个在各项基准领先、具备真实场景零样本泛化能力的双系统导航基础模型 InternVLA-N1。

## 🔥 最新动态

| 时间   | 更新 |
|---------|--------|
| 2025/11 | InternNav v0.2.0 released — added distributed evaluation support for VLN-PE.|
| 2025/10 | Add a [inference-only demo](scripts/notebooks/inference_only_demo.ipynb) of InternVLA-N1. |
| 2025/10 | InternVLA-N1 [technical report](https://internrobotics.github.io/internvla-n1.github.io/static/pdfs/InternVLA_N1.pdf) is released. Please check our [homepage](https://internrobotics.github.io/internvla-n1.github.io/). |
| 2025/09 | Real-world deployment code of InternVLA-N1 released. Upload 3D printing [files](assets/3d_printing_files/go2_stand.STEP) for Unitree Go2. |
| 2025/07 | Hosting the 🏆 IROS 2025 Grand Challenge (see updates at [official website](https://internrobotics.shlab.org.cn/challenge/2025/)) |
| 2025/07 | InternNav v0.1.1 released |

## 📋 目录
- [🏠 介绍](#-介绍)
- [🔥 最新动态](#-最新动态)
- [📚 快速开始](#-快速开始)
- [📦 基准与模型库概览](#-基准与模型库概览)
- [🔧 自定义与拓展](#-自定义与拓展)
- [👥 贡献](#-贡献)
- [🔗 引用](#-引用)
- [📄 许可证](#-许可证)
- [👏 致谢](#-致谢)

## 📚 快速开始

安装、训练与评估等快速上手步骤请参阅[文档](https://internrobotics.github.io/user_guide/internnav/quick_start/index.html)。

## 🤖 真实场景部署（ROS 1）

我们提供了兼容 ROS 1 的客户端，用于在实体机器人上运行 InternVLA-N1 闭环导航流程。

- **依赖：** Ubuntu 20.04 + ROS Noetic（需 `rospy`、`message_filters`、`cv_bridge`、`sensor_msgs`），以及 `requirements/realworld.txt` 中的 Python 依赖。可通过 `apt` 安装 ROS（示例：`sudo apt install ros-noetic-desktop-full ros-noetic-cv-bridge ros-noetic-image-transport`），随后 source 工作空间并执行 `pip install -r requirements/realworld.txt`。
- **话题：** ROS 1 节点默认订阅 `/camera/camera/color/image_raw`（RGB）、`/camera/camera/aligned_depth_to_color/image_raw`（深度）、`/odom_bridge`（里程计），并向 `/cmd_vel_bridge` 发布速度指令。如需适配其他话题名称，可在 launch 文件中进行重映射。
- **服务器交互：** 客户端将 RGB/Depth 帧发送至 HTTP 服务器（默认端口 `5801`，路由 `/eval_dual`）。请确保机器人网络能访问服务器。
- **运行方式：** 构建 catkin 工作空间后，可使用 `rosrun internnav http_internvla_client_ros1.py` 启动节点，或编写包含节点与话题映射的 `roslaunch` 文件。
- **容器建议（可选）：** 若需快速部署，可在 Ubuntu 20.04 基础镜像安装 ROS Noetic：`apt-get install ros-noetic-desktop-full` → `source /opt/ros/noetic/setup.bash` → 安装 `python3-catkin-tools`、`cv-bridge` 与项目依赖。将本仓库挂载进容器，构建工作空间后按上述方式运行节点。

## 📦 基准与模型库概览

### 数据集与基准

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>System2 (VLN-CE)</b>
      </td>
      <td>
         <b>System1 (VN)</b>
      </td>
      <td>
         <b>Whole-system (VLN)</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">VLN-CE R2R</a></li>
            <li align="left"><a href="">VLN-CE RxR</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">Cluttered Envs</a></li>
            <li align="left"><a href="">GRScenes-100</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">VLN-CE</a></li>
            <li align="left"><a href="">VLN-PE</a></li>
         </ul>
      </td>
   </tbody>
</table>

### 模型

<table align="center">
  <tbody>
    <tr align="center" valign="bottom">
      <td>
         <b>System2 (VLN-CE)</b>
      </td>
      <td>
         <b>System1 (VN)</b>
      </td>
      <td>
         <b>Whole-system (VLN)</b>
      </td>
   </tr>
   <tr align="center" valign="top">
      <td>
         <ul>
            <li align="left"><a href="">StreamVLN</a></li>
            <li align="left"><a href="">InternVLA-N1-Preview (S2)</a></li>
            <li align="left"><a href="">InternVLA-N1 (S2)</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">DD-PPO</a></li>
            <li align="left"><a href="">iPlanner</a></li>
            <li align="left"><a href="">ViPlanner</a></li>
            <li align="left"><a href="">GNM</a></li>
            <li align="left"><a href="">ViNT</a></li>
            <li align="left"><a href="">NoMad</a></li>
            <li align="left"><a href="">NavDP</a></li>
         </ul>
      </td>
      <td>
         <ul>
            <li align="left"><a href="">Seq2Seq</a></li>
            <li align="left"><a href="">CMA</a></li>
            <li align="left"><a href="">RDP</a></li>
            <li align="left"><a href="">InternVLA-N1-Preview</a></li>
            <li align="left"><a href="">InternVLA-N1</a></li>
         </ul>
      </td>
   </tbody>
</table>

### 基准结果

#### VLN-CE 任务
| Model  | Dataset/Benchmark | NE | OS | SR | SPL | Download |
| ------ | ----------------- | -- | -- | --------- |  -- | --------- |
| `InternVLA-N1 (S2)` | R2R | 4.89 | 60.6 | 55.4 | 52.1| [Model](https://huggingface.co/InternRobotics/InternVLA-N1-S2) |
| `InternVLA-N1` | R2R | **4.83** | **63.3** | **58.2** | **54.0** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |
| `InternVLA-N1 (S2)` | RxR | 6.67 | 56.5 | 48.6 | 42.6 | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-S2) |
| `InternVLA-N1` | RxR | **5.91** | **60.8** | **53.5** | **46.1** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |
| `InternVLA-N1-Preview (S2)` | R2R | 5.09 | 60.9 | 53.7 | 49.7 | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview-S2) |
| `InternVLA-N1-Preview` | R2R | **4.76** | **63.4** | **56.7** | **52.6** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |
| `InternVLA-N1-Preview (S2)` | RxR | 6.39 | 60.1 | 50.5 | 43.3 | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview-S2) |
| `InternVLA-N1-Preview` | RxR | **5.65** | **63.2** | **53.5** | **45.7** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |

#### VLN-PE 任务
| Model  | Dataset/Benchmark | NE | OS | SR | SPL | Download |
| ------ | ----------------- | -- | -- | -- | --- | --- |
| `Seq2Seq` | Flash | 8.27 | 43.0 | 15.7 | 9.7 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `CMA` | Flash | 7.52 | 45.0 | 24.4 | 18.2 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `RDP` | Flash | 6.98 | 42.5 | 24.9 | 17.5 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `InternVLA-N1-Preview` | Flash | **4.21** | **68.0** | **59.8** | **54.0** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |
| `InternVLA-N1` | Flash | **4.13** | **67.6** | **60.4** | **54.9** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |
| `Seq2Seq` | Physical | 7.88 | 28.1 | 15.1 | 10.7 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `CMA` | Physical | 7.26 | 31.4 | 22.1 | 18.6 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `RDP` | Physical | 6.72 | 36.9 | 25.2 | 17.7 | [Model](https://huggingface.co/InternRobotics/VLN-PE) |
| `InternVLA-N1-Preview` | Physical | **5.31** | **49.0** | **42.6** | **35.8** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1-Preview) |
| `InternVLA-N1` | Physical | **4.73** | **56.7** | **50.6** | **43.3** | [Model](https://huggingface.co/InternRobotics/InternVLA-N1) |

#### 视觉导航任务 - PointGoal Navigation
| Model  | Dataset/Benchmark | SR | SPL | Download |
| ------ | ----------------- | -- | -- | --------- |
| `iPlanner` | ClutteredEnv | 84.8 | 83.6 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `ViPlanner` | ClutteredEnv | 72.4 | 72.3 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `InternVLA-N1 (S1)` | ClutteredEnv | **89.8** | **87.7** | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `iPlanner` | InternScenes | 48.8 | 46.7 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `ViPlanner` | InternScenes | 54.3 | 52.5 | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |
| `InternVLA-N1 (S1)` | InternScenes | **65.7** | **60.7** | [Model](https://github.com/InternRobotics/NavDP?tab=readme-ov-file#%EF%B8%8F-installation-of-baseline-library) |



**说明：**
- VLN-CE RxR 基准与 StreamVLN 即将上线支持。

## 🔧 自定义与拓展

更高级的使用方式（如自定义数据集、模型与实验配置）请参阅[教程](https://internrobotics.github.io/user_guide/internnav/tutorials/index.html)。

## 👥 贡献

欢迎通过提交 Issue、修复框架中的 bug、适配/新增策略与数据等方式参与贡献，具体流程请参考[贡献指南]()。

**提示：** 我们欢迎分享模型在您自有环境中的零样本表现与改进需求，会择优与用户协作推进。

## 🔗 引用

如果本项目对您的研究或产品有帮助，请引用：

```bibtex
@misc{internnav2025,
    title = {{InternNav: InternRobotics'} open platform for building generalized navigation foundation models},
    author = {InternNav Contributors},
    howpublished={\url{https://github.com/InternRobotics/InternNav}},
    year = {2025}
}
```

如使用了特定的预训练模型或基准，请同时引用相关原始论文。下方提供了项目相关的 BibTex 条目。

<details><summary>相关工作 BibTex</summary>

```BibTex
@misc{internvla-n1,
    title = {{InternVLA-N1: An} Open Dual-System Navigation Foundation Model with Learned Latent Plans},
    author = {InternNav Team},
    year = {2025},
    booktitle={arXiv},
}
@inproceedings{vlnpe,
  title={Rethinking the Embodied Gap in Vision-and-Language Navigation: A Holistic Study of Physical and Visual Disparities},
  author={Wang, Liuyi and Xia, Xinyuan and Zhao, Hui and Wang, Hanqing and Wang, Tai and Chen, Yilun and Liu, Chengju and Chen, Qijun and Pang, Jiangmiao},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
@misc{streamvln,
    title = {StreamVLN: Streaming Vision-and-Language Navigation via SlowFast Context Modeling},
    author = {Wei, Meng and Wan, Chenyang and Yu, Xiqian and Wang, Tai and Yang, Yuqiang and Mao, Xiaohan and Zhu, Chenming and Cai, Wenzhe and Wang, Hanqing and Chen, Yilun and Liu, Xihui and Pang, Jiangmiao},
    booktitle={arXiv},
    year = {2025}
}
@misc{navdp,
    title = {NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance},
    author = {Wenzhe Cai, Jiaqi Peng, Yuqiang Yang, Yujian Zhang, Meng Wei, Hanqing Wang, Yilun Chen, Tai Wang and Jiangmiao Pang},
    year = {2025},
    booktitle={arXiv},
}
```

</details>


## 📄 许可证

InternNav 代码遵循 [MIT 许可证](LICENSE)。开源的 InternData-N1 数据采用 <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License </a><a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>。其他数据集（如 VLN-CE）保留其各自的分发许可。

## 👏 致谢

- [InternUtopia](https://github.com/InternRobotics/InternUtopia)（原 `GRUtopia`）：闭环评估与 GRScenes-100 数据依赖该框架。
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)：扩散策略实现。
- [LongCLIP](https://github.com/beichenzbc/Long-CLIP)：长文本 CLIP 模型。
- [VLN-CE](https://github.com/jacobkrantz/VLN-CE)：基于 Habitat 的视觉-语言连续环境导航。
- [Qwen2.5-VL](https://github.com/QwenLM/Qwen2.5-VL)：预训练视觉语言基础模型。
- [LeRobot](https://github.com/huggingface/lerobot)：数据格式设计参考了 LeRobot。
