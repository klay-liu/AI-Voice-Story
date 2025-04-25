#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AI有声故事应用

该应用结合LLM与语音合成技术，可以根据用户提供的情节生成故事，
并使用用户自己的声音进行讲述。
"""

import os
import sys
import time
import tempfile
import shutil
from typing import Tuple, List, Optional, Generator

import torch
import torchaudio
import numpy as np
import gradio as gr
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 添加CosyVoice路径到系统路径
COSYVOICE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'CosyVoice')
sys.path.append(COSYVOICE_PATH)
sys.path.append(os.path.join(COSYVOICE_PATH, 'third_party', 'Matcha-TTS'))

# 导入CosyVoice模块
try:
    from cosyvoice.cli.cosyvoice import CosyVoice2
    from cosyvoice.utils.file_utils import load_wav
    print("成功导入CosyVoice模块")
except ImportError as e:
    print(f"错误: 无法导入CosyVoice模块: {e}")
    print(f"请确保已克隆CosyVoice库到: {COSYVOICE_PATH}")
    print("克隆命令: git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git")
    sys.exit(1)

# 全局变量
OPENAI_API_KEY = os.getenv("API_KEY", "lm-studio")
OPENAI_BASE_URL = os.getenv("BASE_URL", "http://localhost:1234/v1")
OPENAI_MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5-14b-instruct-1m")
COSYVOICE_MODEL_PATH = os.path.join(COSYVOICE_PATH, 'pretrained_models', 'CosyVoice2-0.5B')

# 模型实例(延迟初始化)
cosyvoice = None
client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


# ============= 语音合成模块 =============

def load_model():
    """延迟加载CosyVoice2模型，避免在导入时就加载大模型"""
    global cosyvoice
    if cosyvoice is None:
        try:
            print(f"正在加载CosyVoice2模型，路径: {COSYVOICE_MODEL_PATH}")
            cosyvoice = CosyVoice2(
                COSYVOICE_MODEL_PATH,
                load_jit=False,
                load_trt=False,
                fp16=False,
                use_flow_cache=False
            )
            print("CosyVoice2模型加载成功")
        except Exception as e:
            print(f"加载CosyVoice2模型失败: {e}")
            return None
    return cosyvoice


def synthesize_voice(script: str, voice_audio_path: str, prompt_text: str) -> Tuple[int, List[str]]:
    """
    使用CosyVoice 2合成声音。
    
    Args:
        script: 要朗读的故事文本
        voice_audio_path: 参考声音音频文件的路径
        prompt_text: 参考音频的文本内容
        
    Returns:
        Tuple of (status_code, [list_of_audio_paths])
    """
    try:
        # 加载模型
        model = load_model()
        if model is None:
            return (500, ["无法加载CosyVoice2模型"])
        
        # 加载参考音频
        prompt_speech_16k = load_wav(voice_audio_path, 16000)
        
        # 将脚本分段
        # 按句号、问号、感叹号分割，支持中英文标点
        sentences = []
        current = ""
        for char in script:
            current += char
            if char in ["。", "！", "？", ".", "!", "?"]:
                if current.strip():
                    sentences.append(current.strip())
                current = ""
        if current.strip():
            sentences.append(current.strip())
        
        # 为了方便管理生成的文件，创建一个临时目录
        temp_dir = tempfile.mkdtemp()
        output_paths = []
        
        # 生成音频
        for idx, sentence in enumerate(sentences):
            # 使用zero_shot模式合成
            results = model.inference_zero_shot(
                sentence, prompt_text, prompt_speech_16k, stream=False
            )
            for i, result in enumerate(results):
                output_path = os.path.join(temp_dir, f'story_part_{idx}_{i}.wav')
                torchaudio.save(output_path, result['tts_speech'], model.sample_rate)
                output_paths.append(output_path)
        
        return (200, output_paths)
    
    except Exception as e:
        import traceback
        error_msg = f"声音合成时出错: {str(e)}\n{traceback.format_exc()}"
        return (500, [error_msg])


def combine_audio_files(file_paths: List[str]) -> Optional[str]:
    """将多个音频文件合并为一个"""
    if not file_paths:
        return None
    
    # 创建输出文件路径
    output_path = tempfile.mktemp(suffix='.wav')
    
    # 读取所有音频文件
    waves = []
    sample_rate = None
    for path in file_paths:
        waveform, sr = torchaudio.load(path)
        if sample_rate is None:
            sample_rate = sr
        elif sample_rate != sr:
            # 如果采样率不同，需要重采样
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        waves.append(waveform)
    
    # 拼接所有音频
    combined = torch.cat(waves, dim=1)
    
    # 保存合并后的音频
    torchaudio.save(output_path, combined, sample_rate)
    
    return output_path


# ============= 故事生成模块 =============

def generate_script(story_plot: str) -> str:
    """使用LLM生成故事脚本"""
    if not story_plot.strip():
        return "请先输入故事情节！"
    
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages= [
                {
                    "role": "system",
                    "content": "你是一名富有经验的儿童文学作者兼教育专家，专精于用简洁明了、充满趣味的语言编写兼具启发性和娱乐性的故事。"
                },
                {
                    "role": "user",
                    "content": f"请基于提供的故事情节，为孩子们创作一则适合个人讲述的精彩故事：{story_plot}。确保故事不仅能够吸引小朋友的兴趣，还能传递积极正面的信息和价值观。"
                }
            ],
            temperature=0.7,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"生成故事时出错: {str(e)}"


def update_script_status(script: str) -> str:
    """更新故事脚本状态"""
    if not script or script == "请先输入故事情节！" or script.startswith("生成故事时出错"):
        return "请先生成故事"
    return "故事已确认! 现在您可以进入步骤2进行音色克隆"


# ============= 应用流程控制 =============

def process_audio_and_generate(
    audio_path: str,
    prompt_text: str,
    script: str
) -> Generator[Tuple[str, Optional[str]], None, None]:
    """
    统一处理上传或录制的音频，进行音色克隆并生成故事音频。
    """
    synthesis_result = []
    
    try:
        # 1. 输入验证
        yield "检查输入...", None
        if not audio_path or not os.path.exists(audio_path):
            yield "错误：未提供有效音频。", None
            return

        if not prompt_text.strip():
            yield "错误：Prompt 文本不能为空。", None
            return

        script_state = update_script_status(script)
        if "请先生成故事" in script_state:
            yield "错误：故事未准备好。", None
            return

        # 2. 调用语音合成
        yield "正在进行声音合成...", None
        synthesis_status, synthesis_result = synthesize_voice(script, audio_path, prompt_text)
        
        if synthesis_status != 200:
            error_msg = synthesis_result[0] if synthesis_result else "未知合成错误"
            yield f"错误：{error_msg}", None
            return

        audio_segment_paths = synthesis_result
        if not audio_segment_paths:
            yield "警告：未生成音频片段。", None
            return

        # 3. 合并音频片段
        yield f"合成成功 {len(audio_segment_paths)} 个片段，正在合并...", None
        combined_audio_path = combine_audio_files(audio_segment_paths)

        if not combined_audio_path or not os.path.exists(combined_audio_path):
            yield "错误：合并音频失败。", None
            return

        # 4. 成功返回
        yield "故事音频生成成功！", combined_audio_path

    except Exception as e:
        import traceback
        error_msg = f"处理音频时发生意外错误: {str(e)}"
        print(f"{error_msg}\n{traceback.format_exc()}")
        yield f"严重错误: {error_msg}", None

    finally:
        # 5. 清理临时文件
        # 清理合成过程中创建的片段文件目录
        if synthesis_result and isinstance(synthesis_result, list) and synthesis_result:
            temp_dir = os.path.dirname(synthesis_result[0])
            if temp_dir and os.path.exists(temp_dir) and temp_dir.startswith(tempfile.gettempdir()):
                try:
                    shutil.rmtree(temp_dir)
                    print(f"已清理临时目录: {temp_dir}")
                except Exception as cleanup_e:
                    print(f"清理临时目录时出错 {temp_dir}: {cleanup_e}")


# ============= Gradio 界面 =============

def create_gradio_interface():
    """创建Gradio界面"""
    with gr.Blocks(title="AI有声故事", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 🎙️ AI有声故事")
        gr.Markdown("这个应用可以帮您自动生成故事并用您的声音来讲述")

        # 尝试找到图片
        image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets", "reading-story-at-home-together.jpg")
        if os.path.exists(image_path):
            gr.Image(image_path,
                     label="欢迎使用AI有声故事",
                     interactive=False,
                     height=200)

        gr.Markdown("---")

        with gr.Tab("步骤 1: 生成故事"):
            gr.Markdown("""
            ### 使用说明:
            1. 在下面的文本框中输入故事的基本情节。
            2. 点击 **"生成故事"** 按钮。
            3. 检查下方生成故事，如果满意，点击 **"使用生成的内容"** 按钮。
            4. 确认故事后，请切换到 **"步骤 2: 音色克隆与讲述"** 选项卡。
            """)

            story_plot = gr.Textbox(
                label="故事情节",
                placeholder="请输入故事的基本情节，例如：'小兔子和小松鼠在森林里找到了一个神奇的宝箱...'"
            )
            generate_btn = gr.Button("生成故事", variant="primary")
            script_output = gr.Textbox(
                label="生成故事",
                lines=10,
                placeholder="生成故事将显示在这里..."
            )
            confirm_script_btn = gr.Button("使用生成的内容")
            script_status = gr.Textbox(label="故事状态", value="请先生成故事", interactive=False)

            # 连接事件
            generate_btn.click(fn=generate_script, inputs=story_plot, outputs=script_output)
            confirm_script_btn.click(fn=update_script_status, inputs=[script_output], outputs=[script_status])

        with gr.Tab("步骤 2: 音色克隆与讲述"):
            gr.Markdown("""
            ### 使用说明:
            1. 在下方区域，您可以 **上传** 一个包含您声音的音频文件（如 .wav, .mp3），或者直接点击 **麦克风图标** 录制您的声音。
            2. 在 **"Prompt文本"** 框中输入您在音频中说的 **确切** 内容。**这必须与音频内容完全匹配！**
            3. 点击 **"使用提供的音频生成故事"** 按钮。
            4. 等待系统完成音色克隆和故事音频的合成。
            5. 生成完成后，您可以在本页面底部播放或下载最终的故事音频。

            #### 注意:
            * 音频质量越好（清晰、无噪音），合成效果越佳。
            * 音频内容与 **"Prompt文本"** 必须 **严格一致**。
            * 建议音频时长在 **5-15秒** 之间。
            """)

            audio_input = gr.Audio(
                sources=["upload", "microphone"],
                type="filepath",
                label="上传音频文件或点击麦克风录制"
            )
            prompt_text_audio = gr.Textbox(
                label="Prompt文本 (音频中的确切内容)",
                placeholder="请输入您在上传或录制音频中所说的确切内容...",
                lines=3
            )
            gen_btn_process = gr.Button("使用提供的音频生成故事", variant="primary")

            gr.Markdown("---")

            status_output = gr.Textbox(
                label="生成状态",
                placeholder="生成过程的状态将显示在这里...",
                interactive=False
            )
            final_audio = gr.Audio(label="生成的故事音频")

            # 连接事件
            gen_btn_process.click(
                fn=process_audio_and_generate,
                inputs=[audio_input, prompt_text_audio, script_output],
                outputs=[status_output, final_audio]
            )

    return app


# ============= 程序入口 =============

if __name__ == "__main__":
    # 检查CosyVoice是否已经正确导入
    if 'cosyvoice' not in sys.modules:
        print("警告: CosyVoice模块未正确导入，应用可能无法正常工作")
        print(f"请确保已克隆CosyVoice库到: {COSYVOICE_PATH}")
        sys.exit(1)
    
    # 检查模型文件是否存在
    if not os.path.exists(COSYVOICE_MODEL_PATH):
        print(f"警告: CosyVoice模型文件未找到，请确保下载模型到: {COSYVOICE_MODEL_PATH}")
        print("下载命令: git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B")
        print("继续运行，但语音合成功能可能无法正常工作")
    
    # 启动Gradio应用
    app = create_gradio_interface()
    app.launch()