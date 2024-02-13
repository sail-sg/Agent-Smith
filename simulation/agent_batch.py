####  https://github.com/OpenBMB/AgentVerse/blob/main/agentverse/agents/simulation_agent/prisoner_dilemma.py
        # In the conversation agent, three placeholders are supported:
        # - ${agent_name}: the name of the agent
        # - ${env_description}: the description of the environment
        # - ${role_description}: the description of the role of the agent
        # - ${chat_history}: the chat history of the agent
####
active_thought_prompt_template = '''Your environment description contains the following points:[\n{}\n]
Your role description contains the following properties:[\n{}\n]
Your chat history contains the following records:[\n{}\n]
Your album contains the following images:[\n{}\n]'''

active_thought_prompt_q = "Consider your environment description, role description and chat history. Please select an image from your album."

active_action_prompt_template = '''Your environment description contains the following points:[\n{}\n]
Your role description contains the following properties:[\n{}\n]
Your chat history contains the following records:[\n{}\n]'''

active_action_prompt_q = "<image>\nConsider your environment description, role description and chat history. Please ask a simple question about the image."

passive_action_prompt_template = '''Your environment description contains the following points:[\n{}\n]
Your role description contains the following properties:[\n{}\n]
Your chat history contains the following records:[\n{}\n]'''

passive_action_prompt_q = "<image>\nConsider your environment description, role description and chat history. {}"
####
import torch
import numpy as np
from PIL import Image

####
from conversion import SeparatorStyle, Conversation
conv_llava_v1 = Conversation(
    system="A chat between a curious human and an artificial intelligence assistant. "
           "The assistant gives helpful, detailed, and polite answers to the human's questions.",
    roles=("USER", "ASSISTANT"),
    version="v1",
    messages=(),
    offset=0,
    sep_style=SeparatorStyle.TWO,
    sep=" ",
    sep2="</s>",
)

class Agent:
    def __init__(self, 
                 agent_name, 
                 env_description,
                 role_description, 
                 chat_history, album, 
                 max_records, 
                ):

        self.agent_name = agent_name
        self.role_description = role_description

        self.env_description = env_description
        
        self.chat_history = chat_history
        
        self.album = album
        
        self.max_records=max_records

    def generate_question(self):
        ####
        assert(len(self.album)>1)
        
        album_list = [": ".join(i.split("/")[-1].split("_")[0:2]) for i in self.album]
        
        state = conv_llava_v1.copy()
        state.system = state.system+"\n"+active_thought_prompt_template.format("\n".join(self.env_description),
                                                             "\n".join(self.role_description),
                                                             "\n".join(self.chat_history[-self.max_records:]),
                                                             "\n".join(album_list))

        state.append_message(state.roles[0], active_thought_prompt_q)
        state.append_message(state.roles[1], None)
        
        active_thought_prompt = state.get_prompt()

        state = conv_llava_v1.copy()
        state.system = state.system+"\n"+active_action_prompt_template.format("\n".join(self.env_description), 
                                                            "\n".join(self.role_description), 
                                                            "\n".join(self.chat_history[-self.max_records:]),)

        state.append_message(state.roles[0], active_action_prompt_q)
        state.append_message(state.roles[1], None)
        
        active_action_prompt = state.get_prompt()
        
        return active_thought_prompt, active_action_prompt
    
    def generate_response(self, prompt):

        state = conv_llava_v1.copy()
        state.system = state.system+"\n"+passive_action_prompt_template.format("\n".join(self.env_description), 
                                                             "\n".join(self.role_description), 
                                                             "\n".join(self.chat_history[-self.max_records:]),)

        state.append_message(state.roles[0], passive_action_prompt_q.format(prompt))
        state.append_message(state.roles[1], None)
        
        passive_action_prompt = state.get_prompt()

        return passive_action_prompt