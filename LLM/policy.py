import sys

import fire
import gradio as gr
import torch
import transformers
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import PeftModel

import torch.nn.functional as F
import os
import torch.nn as nn
import numpy as np

from critic import Critic
from torch.distributions.categorical import Categorical
import copy

root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(root)

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class LLMAgent(nn.Module):
    def __init__(self, normalization_mode='token', load_path=None, load_8bit=False):
        super().__init__()

        self.load_8bit = load_8bit
        self.base_model = 'Neko-Institute-of-Science/LLaMA-7B-HF'
        self.lora_r = 8
        self.lora_alpha = 16
        # self.lora_dropout = 0.05
        self.lora_dropout = 0
        self.lora_target_modules = ["q_proj", "v_proj", ]

        assert (
            self.base_model
        ), "Please specify a --base_model, e.g. --base_model='decapoda-research/llama-7b-hf'"

        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        try:
            if torch.backends.mps.is_available():
                self.device = "mps"
        except:  # noqa: E722
            pass

        self.normalization_mode = normalization_mode

        self.tokenizer = LlamaTokenizer.from_pretrained(self.base_model)
        self.tokenizer.pad_token_id = (
            0  # unk. we want this to be different from the eos token
        )

        self.llama = self._init_llama()

        if load_path:
            self.load(load_path)
        else:
            self.actor = self._init_actor().to(self.device)
            self.critic = self._init_critic().to(self.device)

    def _init_llama(self):
        model = LlamaForCausalLM.from_pretrained(
            self.base_model,
            torch_dtype=torch.float16,
            load_in_8bit=self.load_8bit,
            device_map="auto",
            cache_dir=os.path.join(root, 'weights/llama')
        )

        if not self.load_8bit:
            model.half().to(self.device)
        else:
            model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

        return model

    def _init_actor(self, lora_weights=None):
        if lora_weights is None:
            config = LoraConfig(
                r=self.lora_r,
                lora_alpha=self.lora_alpha,
                target_modules=self.lora_target_modules,
                lora_dropout=self.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(self.llama, config)

            model.print_trainable_parameters()

            old_state_dict = model.state_dict
            model.state_dict = (
                lambda self, *_, **__: get_peft_model_state_dict(
                    self, old_state_dict()
                )
            ).__get__(model, type(model))
        else:
            model = PeftModel.from_pretrained(
                self.llama,
                lora_weights,
                torch_dtype=torch.float16,
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            model = torch.compile(model)

        if not self.load_8bit:
            model.half()
        else:
            model = prepare_model_for_int8_training(model, use_gradient_checkpointing=True)

        return model

    def _init_critic(self, critic_weights=None):
        critic = Critic(self.actor, self.tokenizer)
        if critic_weights is not None:
            critic.v_head.load_state_dict(torch.load(critic_weights, map_location="cpu"))
        return critic

    def save(self, epoch, exp_path):
        print("save model")
        exp_path = os.path.join(exp_path, "epoch_{:04d}".format(epoch))

        os.makedirs(exp_path, exist_ok=True)
        # save lora
        self.actor.save_pretrained(exp_path)
        # save critic
        # torch.save(self.critic.v_head.state_dict(), os.path.join(exp_path, "critic.pth"))

    def load(self, exp_path):
        print("load model")
        lora_weights = exp_path
        # critic_weights = os.path.join(exp_path, "critic.pth")
        self.actor = self._init_actor(lora_weights).to(self.device)
        # self.critic = self._init_critic(critic_weights).to(self.device)

    def get_value(self, x):
        if type(x) != list:
            x = [self.obs2text(o)["prompt"] for o in x]

        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with self.actor.disable_adapter():
            value = self.critic(input_ids, attention_mask=attention_mask)
        return value

    def get_action_and_value(self, obs, action=None, is_warmup=False, return_value=True):
        text_obs = [self.obs2text(o) for o in obs]
        prompt = [o["prompt"] for o in text_obs]

        action_list = [o["action"] for o in text_obs]
        action_ids = [[self.template2action[item] for item in env] for env in action_list]

        prompt_nums = len(prompt)
        action_nums = [len(item) for item in action_list]

        sequence = []
        for p, ac in zip(prompt, action_list):
            sequence += [p + " " + a for a in ac]

        inputs = self.tokenizer(sequence, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)

        attention_mask = inputs["attention_mask"].to(self.device)

        if is_warmup:
            with torch.no_grad():
                outputs = self.actor(input_ids, attention_mask=attention_mask)
        else:
            outputs = self.actor(input_ids, attention_mask=attention_mask)

        action_list = [item for sublist in action_list for item in sublist]
        self.action_list_ids = self.tokenizer(action_list, return_tensors="pt", padding=True)

        self.action_list_length = torch.sum(self.action_list_ids["attention_mask"], dim=-1) - 1  # delete first token

        sequence_length = torch.sum(attention_mask, dim=-1)
        action_index = [[end - start, end] for start, end in zip(self.action_list_length, sequence_length)]

        logits = torch.log_softmax(outputs.logits, dim=-1)

        logits = logits[:, :-1, :]
        input_ids = input_ids[:, 1:]
        gen_logits = torch.gather(logits, 2, input_ids[:, :, None]).squeeze(-1)

        slices = [gen_logits[i, start - 1:end - 1] for i, (start, end) in enumerate(action_index)]

        action_logits = torch.stack([torch.sum(s) for s in slices])

        if self.normalization_mode == 'token':
            action_logits = action_logits / self.action_list_length.to(self.device)
        elif self.normalization_mode == 'word':
            action_word_num = torch.tensor([len(action.split()) for action in action_list]).to(self.device)
            action_logits = action_logits / action_word_num
        elif self.normalization_mode == 'sum':
            action_logits = action_logits
        else:
            assert 1 == 2

        actions = []
        log_probs = []
        entroy = []

        for i in range(prompt_nums):
            logits = action_logits[sum(action_nums[:i]):sum(action_nums[:i + 1])].reshape(-1, action_nums[i]).float()

            probs = Categorical(logits=logits)

            if action is None:
                cur_action = probs.sample()[0]
                cur_action = cur_action.view(-1)
                real_action = torch.tensor([action_ids[i][cur_action.item()]], dtype=torch.int32).to(self.device)
            else:
                real_action = action[i].view(-1)
                cur_action = torch.tensor([action_ids[i].index(real_action.item())], dtype=torch.int32).to(self.device)

            actions.append(real_action)
            log_probs.append(probs.log_prob(cur_action))
            entroy.append(probs.entropy())

        action = torch.cat(actions)
        log_probs = torch.cat(log_probs)
        entroy = torch.cat(entroy)

        if return_value:
            return action, log_probs, entroy, self.get_value(prompt)
        else:
            return action, log_probs, entroy, None

    # TODO: Rewrite this function to use our prompts
    def obs2text(self, obs):
        text = ""

        # 假设 obs 中的索引和字段如下对应（你可以根据需要调整这些索引）
        clickable = obs[0]
        scrollable = obs[1]
        checkable = obs[2]
        long_clickable = obs[3]
        editable = obs[4]
        rotatable = obs[5]
        searchable = obs[6]
        swipeable = obs[7]




        actionable = clickable or scrollable or checkable or long_clickable or editable or searchable or swipeable

        object_text = ""
        action_list = []

        if actionable:
            action_phrases = []
            action_phrases.append("press any key")
            action_list.append(8)

            if editable:
                action_phrases.append("edit the screen")
                action_list.append(0)
            if searchable:
                action_phrases.append("search the screen")
                action_list.append(16)
            if editable and searchable:
                action_phrases.append("edit the screen and search the screen")
                action_list.append(17)

            if clickable :
                action_phrases.append("click on the screen")
                action_list.append(1)
                action_list.append(2)
            if swipeable:
                action_phrases.append("swipe the screen")
                action_list.append(3)

            if scrollable:
                action_phrases.append("scroll the screen up or down")
                action_list.append(4)
                action_list.append(5)
                action_list.append(6)
                action_list.append(7)
            if rotatable:
                action_phrases.append("rotate the screen")
                action_list.append(12)
                action_list.append(13)

            object_text = "You can " + ", ".join(action_phrases) + "."

        else:
            action_phrases = []
            object_text = "There are no available interactions on the screen at the moment."
            action_phrases.append("spawn a new event")
            action_list.append(10)  # 对应 "spawn event"

            action_phrases.append("kill the app")
            action_list.append(11)  # 对应 "kill app event"

            action_phrases.append("fresh reinstall the app")
            action_list.append(14)  # 对应 "fresh reinstall app"

            action_phrases.append("kill and restart the app")
            action_list.append(15)  # 对应 "kill and restart app"

            action_phrases.append("exit the app")
            action_list.append(18)

            object_text = "But you can " + ", ".join(action_phrases) + "."

        text += object_text

        # template for target
        target_template = " In order to complete your goal, "
        text += target_template

        # template for next step
        next_step_text = "your next step is to..."
        text += next_step_text

        self.action_template = [
            "edit",  # 对应 SetTextEvent 0
            "click",  # 对应 TouchEvent 1
            "long click",  # 对应 LongTouchEvent 2
            "swipe",  # 对应 SwipeEvent 3
            "scroll up",  # 对应 ScrollEvent，方向为 UP 4
            "scroll down",  # 对应 ScrollEvent，方向为 DOWN 5
            "scroll left",  # 对应 ScrollEvent，方向为 LEFT 6
            "scroll right",  # 对应 ScrollEvent，方向为 RIGHT 7
            "key event",  # 对应 KeyEvent 8
            "intent event",  # 对应 IntentEvent 9
            "spawn event",  # 对应 SpawnEvent 10
            "kill app event",  # 对应 KillAppEvent 11
            "rotate device to landscape",  # 对应 RotateDeviceToLandscapeEvent 12
            "rotate device to portrait",  # 对应 RotateDeviceToPortraitEvent 13
            "fresh reinstall app",  # 对应 ReInstallAppEvent 14
            "kill and restart app",  # 对应 KillAndRestartAppEvent 15
            "search",  # 对应 SearchEvent 16
            "set text and search",  # 对应 SetTextAndSearchEvent 17
            "exit",  # 对应 ExitEvent 18
        ]

        self.template2action = {
            k: i for i, k in enumerate(self.action_template)
        }

        actions = [self.action_template[i] for i in action_list]

        return {"prompt": text, "action": actions}

