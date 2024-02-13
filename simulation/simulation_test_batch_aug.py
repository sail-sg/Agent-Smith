####
import os
from io import BytesIO
####
import torch
import random
import numpy as np

def set_seeds(seed):    
    random.seed(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
set_seeds(42)
####
from accelerate import Accelerator

from accelerate.utils import gather_object
from transformers import AutoModelForCausalLM, AutoTokenizer
from statistics import mean
import torch, time, json

accelerator = Accelerator()
####
import json
from PIL import Image

import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from transformers import CLIPProcessor, CLIPModel

####
import argparse

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--attack_image", type=str, default="", help="")
    parser.add_argument("--num_attacks", type=int, default=1, help="")

    parser.add_argument("--high", action='store_true', default=False, help="")
    parser.add_argument("--num_agents", type=int, default=64, help="number of agents")
    parser.add_argument("--num_rounds", type=int, default=32, help="number of rounds")
    parser.add_argument("--max_records", type=int, default=3, help="")
    parser.add_argument("--album_length", type=int, default=10, help="")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="")

    # parameters related to train and evaluation
    parser.add_argument("--agent_data", type=str, default="./data/million_villagers_1024_test.json", help="")
    parser.add_argument("--album_data", type=str, default="./data/album_pool/{}", help="")

    # parameters related to VLM
    parser.add_argument("--vlm", type=str, default="llava-hf/llava-1.5-7b-hf", help="vlm model path")
    parser.add_argument("--clip", type=str, default="openai/clip-vit-large-patch14", help="vlm model path")

    # 
    parser.add_argument("--seed", type=int, default=42, help="")

    #### 
    parser.add_argument("--prob_random_flip", type=float, default=0.5, help="")
    parser.add_argument("--enable_random_resize", type=str, default="yes", help="")
    parser.add_argument("--prob_random_jpeg", type=float, default=0.5, help="")
    ####

    args = parser.parse_args()
    return args
####

def main(args):
    ####
    if args.attack_image=="":
        print("error")
        return
    ####
    if args.high==True:
        from agent_high_batch import Agent
    else:
        from agent_batch import Agent
    ####
    
    model_id = args.vlm
    torch_dtype = torch.bfloat16
    
    processor = AutoProcessor.from_pretrained(model_id)
    
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id, 
        device_map={"": accelerator.device},
        torch_dtype=torch_dtype, 
        low_cpu_mem_usage=True,
    )
    
    model.eval()
    
    clip_model_id = args.clip
    clip_processor = CLIPProcessor.from_pretrained(clip_model_id) # 224
    clip_model = CLIPModel.from_pretrained(clip_model_id, 
                                           device_map={"": accelerator.device},
                                           torch_dtype=torch_dtype,
                                           low_cpu_mem_usage=True,
                                          )
    
    clip_model.eval()
    #### 
    print('perturb')
    processor_ = AutoProcessor.from_pretrained(model_id)
    clip_processor_ = CLIPProcessor.from_pretrained(clip_model_id) # 224
    
    import torchvision
    
    from transformers.image_transforms import (
    convert_to_rgb
    )
    from transformers.image_utils import (
    make_list_of_images,   
    )
    
    def convertToJpeg(im, quality=75):
        with BytesIO() as f:
            im.save(f, format='JPEG', quality=quality)
            f.seek(0)
            im = Image.open(f)
            im.load()
            return im
            
    def llava_warpper(images, **kwargs):
        images = make_list_of_images(images)
        ####
        images = [convert_to_rgb(image) for image in images]
        #### random jpeg
        rand = random.random()
        if rand>(1-args.prob_random_jpeg):
            quality = 75
            images = [convertToJpeg(image, quality) for image in images]
        else:
            pass
        #### random flip
        random_flip = torchvision.transforms.RandomHorizontalFlip(p=args.prob_random_flip)
        images = [random_flip(i) for i in images]
        #### random resize
        if args.enable_random_resize=="yes":
            random_resize = random.randint(224, 448)
            images = [image.resize((random_resize, random_resize)) for image in images]
        ####
        return processor_.image_processor.preprocess(images, **kwargs)
        
    def clip_warpper(images, **kwargs):
        images = make_list_of_images(images)
        ####
        images = [convert_to_rgb(image) for image in images]
        #### random jpeg
        rand = random.random()
        if rand>(1-args.prob_random_jpeg):
            quality = 75
            images = [convertToJpeg(image, quality) for image in images]
        else:
            pass
        #### random flip
        random_flip = torchvision.transforms.RandomHorizontalFlip(p=args.prob_random_flip)
        images = [random_flip(i) for i in images]
        #### random resize
        if args.enable_random_resize=="yes":
            random_resize = random.randint(224, 448)
            images = [image.resize((random_resize, random_resize)) for image in images]
        ####
        return clip_processor_.image_processor.preprocess(images, **kwargs)
    
    processor.image_processor.preprocess = llava_warpper
    clip_processor.image_processor.preprocess = clip_warpper
    ####
    
    ####
    selected_keys = ["Name", 
                     "Species", 
                     "Gender", 
                     "Personality", 
                     "Subtype",
                     "Hobby",
                     "Birthday",
                     "Catchphrase",
                     "Favorite Song",
                     "Favorite Saying",
                     "Style 1",
                     "Style 2",
                     "Color 1",
                     "Color 2",
                    ]
    ####
    from unidecode import unidecode
    set_seeds(args.seed)
    
    
    with open(args.agent_data, 'r') as handle:
        villagers = json.load(handle)
        
    print(len(villagers))
    
    sampled_villagers = np.random.choice(villagers, args.num_agents, replace=False)
    ####
    agent_list = []
    
    for villager in sampled_villagers:
        ####
        role_description = ["{}: {}".format(key, villager[key]) for key in selected_keys]
        ####
        env_description = []
        ####
        album = [args.album_data.format(i) for i in villager["Furniture List"].split(";")]
    
        if len(album)>=args.album_length:
            album = album[0:args.album_length]
        else:
            album = album + list(np.random.choice(album, args.album_length-len(album)))
        
        chat_history = []
        
        agent = Agent(
            villager["Name"], env_description,
            role_description, chat_history, album, args.max_records)
    
        agent_list.append(agent)
    ####
    attack_image = args.attack_image
    
    for i in range(args.num_attacks):
        agent_list[i].album.append(attack_image)
        agent_list[i].album.pop(0)
    #### environment
    def pop_random(lst):
        idx = random.randrange(0, len(lst))
        return lst.pop(idx)
    
    set_seeds(args.seed)
    
    round_list = []
    for round in range(args.num_rounds):
        # print("#### round: ", round)
        conversations_list = []
        lst = list(range(len(agent_list)))
        while len(lst)>=2:
            rand1 = pop_random(lst)
            rand2 = pop_random(lst)
    
            agent = agent_list[rand1]
            paired_agent = agent_list[rand2]
            #### 
            env_description = ["{} is chatting with {}.".format(agent.agent_name, paired_agent.agent_name)] 
            agent.env_description = env_description
            paired_agent.env_description = env_description
            ####
            conversations_list.append((agent, paired_agent))
    
        # sync GPUs
        accelerator.wait_for_everyone()
    
        with accelerator.split_between_processes(list(range(len(conversations_list))), apply_padding=True) as conversations_idx:
            
            my_dict_list = []
            ####
            if args.max_records==3:
                batch_size = 8
            elif args.max_records==6:
                batch_size = 8
            elif args.max_records==9:
                batch_size = 4
            elif args.max_records==12:
                batch_size = 2
            else:
                batch_size = 8
            ####
    
            active_thought_prompt_list = []
            active_action_prompt_list = []
            album_list = []
            for idx in conversations_idx:
                ####
                (agent, _) = conversations_list[idx]
                ####
                active_thought_prompt, active_action_prompt = agent.generate_question()
                active_thought_prompt_list.append(active_thought_prompt)
                active_action_prompt_list.append(active_action_prompt)
                ####
                album_list.append(agent.album)
                ####
    
            #### Step one
            batches=[active_thought_prompt_list[i:i + batch_size] for i in range(0, len(active_thought_prompt_list), batch_size)] 
    
            active_thought_output_list = []
            for batch in batches:
                inputs = processor(batch, return_tensors='pt', padding=True)
                for key in inputs:
                    if key!='pixel_values':
                        inputs[key] = inputs[key].to(accelerator.device)
                
                outputs = model.generate(**inputs, 
                                     max_new_tokens=77, # !!!!
                                     do_sample=False)
                
                active_thought_output_list += processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
            #### Step two: RAG
            raw_image_list = []
    
            batches=[(active_thought_output_list[i:i + batch_size], 
                      album_list[i:i + batch_size]) for i in range(0, len(active_thought_output_list), batch_size)] 
    
            similarity_list = []
            for batch, albums in batches:
                clip_text_input = clip_processor.tokenizer(
                        batch,
                        padding="max_length",
                        max_length=77,
                        truncation=True,
                        return_tensors="pt",
                    ).to(accelerator.device) # fix
                
                clip_image_input = clip_processor.image_processor.preprocess(
                    [Image.open(i) for album in albums for i in album],
                    return_tensors='pt').pixel_values
                
                clip_image_input = clip_image_input.to(accelerator.device)   
    
                with torch.no_grad():
                    text_embeddings_clip = clip_model.get_text_features(clip_text_input.input_ids, 
                                                                        clip_text_input.attention_mask) # fix
                    
                    text_embeddings_clip = text_embeddings_clip / text_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
    
                    bsz, hidden = text_embeddings_clip.size()
                    
                    image_embeddings_clip = clip_model.get_image_features(clip_image_input)
                    image_embeddings_clip = image_embeddings_clip / image_embeddings_clip.norm(p=2, dim=-1, keepdim=True)
        
                    image_embeddings_clip = image_embeddings_clip.reshape(bsz, -1, hidden)
    
                    similarity = torch.bmm(text_embeddings_clip.unsqueeze(1), image_embeddings_clip.permute(0, 2, 1)) # bsz, 1, 10
                    similarity = similarity.float().cpu().numpy() # !!!
                                
                for i in range(bsz):
                    selected_image_idx = np.argmax(similarity[i])
    
                    similarity_list.append(similarity[i])
                    
                    raw_image = albums[i][selected_image_idx]
                    raw_image_list.append(raw_image)
                    
            #### Step three
            batches=[(active_action_prompt_list[i:i + batch_size], 
                      raw_image_list[i:i + batch_size]) for i in range(0, len(active_action_prompt_list), batch_size)] 
            
            active_action_output_list = []
    
            for batch, images in batches:
                inputs = processor(batch, 
                                   [Image.open(i) for i in images],
                                   return_tensors='pt', padding=True).to(accelerator.device, torch_dtype)  
                
                outputs = model.generate(**inputs, 
                                     max_new_tokens=args.max_new_tokens, # !!!!
                                     do_sample=False)
                active_action_output_list += processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            #### Step four
            passive_action_prompt_list = []
            for idx, prompt in zip(conversations_idx, active_action_output_list):
                ####
                (_, paired_agent) = conversations_list[idx]
                ####
                passive_action_prompt = paired_agent.generate_response(prompt)
                passive_action_prompt_list.append(passive_action_prompt)
            
            batches=[(passive_action_prompt_list[i:i + batch_size], 
                      raw_image_list[i:i + batch_size]) for i in range(0, len(passive_action_prompt_list), batch_size)] 
            
            passive_action_output_list = []
    
            for batch, images in batches:
                inputs = processor(batch, 
                                   [Image.open(i) for i in images],
                                   return_tensors='pt', padding=True).to(accelerator.device, torch_dtype)  
                
                outputs = model.generate(**inputs, 
                                     max_new_tokens=args.max_new_tokens, # !!!!
                                     do_sample=False)
                passive_action_output_list += processor.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
            ####
            for index, idx in enumerate(conversations_idx):
                (agent, paired_agent) = conversations_list[idx]
    
                active_thought_prompt = active_thought_prompt_list[index]
                active_thought_output = active_thought_output_list[index]
                active_action_prompt = active_action_prompt_list[index]
                active_action_output = active_action_output_list[index]
                passive_action_prompt = passive_action_prompt_list[index]
                passive_action_output = passive_action_output_list[index]
                raw_image = raw_image_list[index]
                similarity = similarity_list[index]
                
                record = "round {}\n{}: {}\n{}: {}".format(round, agent.agent_name,
                                                           active_action_output, 
                                                           paired_agent.agent_name, 
                                                           passive_action_output)
                print(record)
                print()
    
                my_dict = {
                "idx": idx, 
                    
                "round": round,
                "agent": agent.agent_name,
                "paired_agent": paired_agent.agent_name,
                
                "active_thought_prompt": active_thought_prompt,
                "active_thought_output": active_thought_output,
                
                "active_action_prompt": active_action_prompt,
                "active_action_output": active_action_output,
                
                "passive_action_prompt": passive_action_prompt,
                "passive_action_output": passive_action_output,
    
                "record": record,
                "raw_image": raw_image,
    
                "similarity": similarity,
    
                "in_album": attack_image in agent.album,
                }
                my_dict_list.append(my_dict)
                ####
        
        my_dict_list_gathered = gather_object(my_dict_list)
        round_list+=my_dict_list_gathered
        
        ####
        seen = []
        for my_dict in my_dict_list_gathered:
            idx = my_dict["idx"]
            if idx in seen:
                pass
            else:
                seen.append(idx)
                
                record = my_dict["record"]
                raw_image = my_dict["raw_image"]
                            
                (agent, paired_agent) = conversations_list[idx]
                
                agent.chat_history.append(record)
                paired_agent.chat_history.append(record)
                
                paired_agent.album.append(raw_image)
                paired_agent.album.pop(0)
        ####
    
    import pandas as pd
    pd.options.display.max_colwidth = None
    
    if accelerator.is_main_process:
        print(len(round_list))
        df = pd.DataFrame.from_records(round_list)
        df = df.drop_duplicates(subset=["round", "agent", "paired_agent"])

        filename = "./data/{}/{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.csv".format(
            "_".join(args.attack_image.split('/')).replace(".", "_"),
            args.num_attacks,
            args.high, 
            args.num_agents, args.num_rounds, 
            args.max_records, args.album_length, 
            args.seed,
            
            args.prob_random_flip, 
            args.enable_random_resize,
            args.prob_random_jpeg,
        )
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        df.to_csv(filename, index=False)

if __name__ == "__main__":
    args = parse_args()
    print(args)
    main(args)