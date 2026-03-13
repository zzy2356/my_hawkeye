import torch
import numpy as np
import re
from llava.constants import X_TOKEN_INDEX, DEFAULT_X_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_X_token, get_model_name_from_path, KeywordsStoppingCriteria
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType
import pandas as pd

from transformers import logging

logging.set_verbosity_error()
import warnings
import os
from tqdm import tqdm

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers", lineno=1656)
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.set_verbosity_warning()
logging.set_verbosity_error()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def _is_qwen3_vl_model_name(model_name_or_path: str) -> bool:
    normalized = (model_name_or_path or "").lower().replace("-", "").replace("_", "")
    return "qwen3" in normalized and "vl" in normalized


def _strip_media_tokens(text: str) -> str:
    return re.sub(r"<\s*(video|image)\s*>", "", text, flags=re.IGNORECASE).strip()


def _aux_stats_text(pose_feature: torch.Tensor, scene_feature: torch.Tensor) -> str:
    pose_feature = pose_feature.float()
    scene_feature = scene_feature.float()
    return (
        f"pose_mean={pose_feature.mean().item():.4f}; pose_std={pose_feature.std().item():.4f}; "
        f"scene_mean={scene_feature.mean().item():.4f}; scene_std={scene_feature.std().item():.4f}"
    )


def _run_qwen3_vl_inference(model, processor, video_path: str, prompt: str, pose_feature: torch.Tensor,
                            scene_feature: torch.Tensor, max_new_tokens: int = 32) -> str:
    user_text = _strip_media_tokens(prompt)
    user_text = f"{user_text}\nAuxiliary scene stats: {_aux_stats_text(pose_feature, scene_feature)}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(model.device)

    with torch.inference_mode():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)

    trimmed_ids = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)]
    outputs = processor.batch_decode(trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return outputs[0].strip() if len(outputs) > 0 else ""


def main():
    disable_torch_init()

    model_path = 'models/Qwen3-VL-8B-Instruct'
    # model_path = 'LanguageBind/video-llava-7b'
    model_base = None
    # model_base = None
    device = 'cuda'
    load_4bit, load_8bit = False, False
    model_name = get_model_name_from_path(model_path)
    is_qwen3_vl = _is_qwen3_vl_model_name(model_path)
    tokenizer, model, processor, context_len = load_pretrained_model(model_path, model_base, model_name, load_8bit,
                                                                     load_4bit, device=device)
    video_processor = processor.get('video')
    qwen_processor = processor.get('qwen')

    for video_id_folder in tqdm(os.listdir('dataset/vid_split/test')):
        # if video_id_folder == '295_Ekman6_anger_932':
        print(video_id_folder)
        video_id_folder_path = os.path.join('dataset/vid_split/test_new', video_id_folder)
        video_list = []
        name = []
        res = []
        for video in os.listdir(video_id_folder_path):
            video_list.append(video)
        video_list = sorted(video_list, key=lambda x: int(x.split("_")[1].split(".")[0]))

        for video in tqdm(video_list):
            name.append(video)
            video_path = os.path.join(video_id_folder_path, video)
            pose_path = os.path.join('dataset/pose_feat/test/{}'.format(video_id_folder),
                                    'frame_{}.npy'.format(int(video.split('.')[0])))

            pose_feature = torch.from_numpy(np.load(pose_path))
            if pose_feature.size(0) < 5:
                padding_size = 5 - pose_feature.size(0)
                pose_feature_pad = torch.cat((pose_feature, torch.zeros((padding_size, 17, 5))), dim=0)
            else:
                pose_feature_pad = pose_feature[:5, :, :]

            scene_path = os.path.join('dataset/graph_feat/test/{}'.format(video_id_folder),
                                    'frame_{}.npy'.format(int(video.split('.')[0])))

            scene_feature = torch.from_numpy(np.load(scene_path))
            if scene_feature.size(0) < 5:
                padding_size = 5 - scene_feature.size(0)
                scene_feature_pad = torch.cat((scene_feature, torch.zeros((padding_size, 353))), dim=0)
            else:
                scene_feature_pad = scene_feature[:5, :]

            inp = 'Your prompt here'

            if is_qwen3_vl:
                if qwen_processor is None:
                    raise ValueError('Qwen3-VL inference requires processor["qwen"].')
                outputs = _run_qwen3_vl_inference(
                    model=model,
                    processor=qwen_processor,
                    video_path=video_path,
                    prompt=inp,
                    pose_feature=pose_feature_pad,
                    scene_feature=scene_feature_pad,
                    max_new_tokens=16,
                )
            else:
                conv_mode = "llava_v1"
                conv = conv_templates[conv_mode].copy()

                video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
                if type(video_tensor) is list:
                    tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                else:
                    tensor = video_tensor.to(model.device, dtype=torch.float16)

                if type(pose_feature_pad) is list:
                    tensor_pose = [pose.to(model.device, dtype=torch.float16) for pose in pose_feature_pad]
                else:
                    tensor_pose = pose_feature_pad.to(model.device, dtype=torch.float16)

                if type(scene_feature_pad) is list:
                    tensor_scene = [scene.to(model.device, dtype=torch.float16) for scene in scene_feature_pad]
                else:
                    tensor_scene = scene_feature_pad.to(model.device, dtype=torch.float16)
                key = ['video']

                inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()
                input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(
                    0).cuda()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=[tensor, [tensor_pose], [tensor_scene], key],
                        do_sample=True,
                        temperature=0.1,
                        max_new_tokens=16,
                        use_cache=True,
                        stopping_criteria=[stopping_criteria])

                outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
            # print(outputs)
            res.append(outputs)
        df = pd.DataFrame({
            'file': name,
            'output': res
        })
        save_path = 'dataset/saved_result/test_res/{}.csv'.format(video_id_folder)
        df.to_csv(save_path, index=False)

    for video_id_folder in tqdm(os.listdir('dataset/Ucf/Ucfcrime_split')):
        if 'Normal' not in video_id_folder:
            video_id_folder_path = os.path.join('dataset/Ucf/Ucfcrime_split', video_id_folder)
            video_list = []
            name = []
            res = []
            for video in os.listdir(video_id_folder_path):
                video_list.append(video)
            video_list = sorted(video_list, key=lambda x: int(x.split("_")[1].split(".")[0]))

            for video in tqdm(video_list):
                name.append(video)
                video_path = os.path.join(video_id_folder_path, video)

                pose_path = os.path.join('dataset/Ucf/pose_feat/{}'.format(video_id_folder),
                                        'frame_{}.npy'.format(int(video.split('.')[0])))

                pose_feature = torch.from_numpy(np.load(pose_path))
                
                if pose_feature.size(0) < 5:
                    padding_size = 5 - pose_feature.size(0)
                    pose_feature_pad = torch.cat((pose_feature, torch.zeros((padding_size, 17, 5))), dim=0)
                else:
                    pose_feature_pad = pose_feature[:5, :, :]

                scene_path = os.path.join('dataset/Ucf/graph_feat/{}'.format(video_id_folder),
                                    'frame_{}.npy'.format(int(video.split('.')[0])))

                scene_feature = torch.from_numpy(np.load(scene_path))
                if scene_feature.size(0) < 5:
                    padding_size = 5 - scene_feature.size(0)
                    scene_feature_pad = torch.cat((scene_feature, torch.zeros((padding_size, 353))), dim=0)
                else:
                    scene_feature_pad = scene_feature[:5, :]

                inp = 'Your prompt here'

                if is_qwen3_vl:
                    if qwen_processor is None:
                        raise ValueError('Qwen3-VL inference requires processor["qwen"].')
                    outputs = _run_qwen3_vl_inference(
                        model=model,
                        processor=qwen_processor,
                        video_path=video_path,
                        prompt=inp,
                        pose_feature=pose_feature_pad,
                        scene_feature=scene_feature_pad,
                        max_new_tokens=32,
                    )
                else:
                    conv_mode = "llava_v1"
                    conv = conv_templates[conv_mode].copy()

                    video_tensor = video_processor(video_path, return_tensors='pt')['pixel_values']
                    if type(video_tensor) is list:
                        tensor = [video.to(model.device, dtype=torch.float16) for video in video_tensor]
                    else:
                        tensor = video_tensor.to(model.device, dtype=torch.float16)
                        
                    if type(pose_feature_pad) is list:
                        tensor_pose = [pose.to(model.device, dtype=torch.float16) for pose in pose_feature_pad]
                    else:
                        tensor_pose = pose_feature_pad.to(model.device, dtype=torch.float16)

                    if type(scene_feature_pad) is list:
                        tensor_scene = [scene.to(model.device, dtype=torch.float16) for scene in scene_feature_pad]
                    else:
                        tensor_scene = scene_feature_pad.to(model.device, dtype=torch.float16)

                    key = ['video']

                    inp = DEFAULT_X_TOKEN['VIDEO'] + '\n' + inp
                    conv.append_message(conv.roles[0], inp)
                    conv.append_message(conv.roles[1], None)
                    prompt = conv.get_prompt()
                    input_ids = tokenizer_X_token(prompt, tokenizer, X_TOKEN_INDEX['VIDEO'], return_tensors='pt').unsqueeze(
                        0).cuda()
                    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                    keywords = [stop_str]
                    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

                    with torch.inference_mode():
                        output_ids = model.generate(
                            input_ids,
                            images=[tensor, [tensor_pose], [tensor_scene], key],
                            do_sample=True,
                            temperature=0.1,
                            max_new_tokens=32,
                            use_cache=True,
                            stopping_criteria=[stopping_criteria])

                    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
                # print(outputs)
                res.append(outputs)
            df = pd.DataFrame({
                'file': name,
                'output': res
            })
            save_path = 'dataset/saved_result/test_res/{}.csv'.format(video_id_folder)
            df.to_csv(save_path, index=False)


if __name__ == '__main__':
    main()
